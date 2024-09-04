package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sync"
	"time"

	xrand "golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat/distuv"
)

type StatEntry struct {
	startTime time.Time
	latency   int64
	url       string
}

type Stats struct {
	mutex         *sync.Mutex
	entries       []StatEntry
	failure_count uint64
	success_count uint64
}

func (s *Stats) Succeeded(startTime time.Time, latency int64, url string) {
	s.mutex.Lock()
	s.success_count += 1
	s.entries = append(s.entries, StatEntry{
		startTime: startTime,
		latency:   latency,
		url:       url,
	})
	s.mutex.Unlock()
}

func (s *Stats) Failed(startTime time.Time, latency int64, url string) {
	s.mutex.Lock()
	s.failure_count += 1
	// s.latencies = append(s.latencies, latency)
	s.mutex.Unlock()
}

func (s *Stats) Mean() (mean float64) {
	s.mutex.Lock()
	var sum int64
	for _, entry := range s.entries {
		sum += entry.latency
	}
	mean = float64(sum) / float64(1e3*len(s.entries))
	s.mutex.Unlock()
	return
}

func (s *Stats) OutputLatencies(path string) {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	log.Printf("Writing latencies at %s\n", path)
	f, err := os.Create(path)
	if err != nil {
		log.Fatal(err)
	}
	for _, entry := range s.entries {
		// output in ms
		f.WriteString(fmt.Sprintf("%f\n", float64(entry.latency)/1e3))
	}
	if err := f.Close(); err != nil {
		log.Fatal(err)
	}
}

func (s *Stats) OutputStartTimes(t time.Time, path string) {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	log.Printf("Writing start times at %s\n", path)
	f, err := os.Create(path)
	if err != nil {
		log.Fatal(err)
	}
	data := make([]float64, len(s.entries))
	for i, entry := range s.entries {
		data[i] = float64(entry.startTime.Sub(t).Microseconds()) / 1e6
	}
	for _, d := range data {
		f.WriteString(fmt.Sprintf("%f\n", d))
	}
	if err := f.Close(); err != nil {
		log.Fatal(err)
	}
}

func (s *Stats) Output(t time.Time, path string) {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	log.Printf("Writing stats at %s\n", path)
	f, err := os.Create(path)
	if err != nil {
		log.Fatal(err)
	}
	writer := csv.NewWriter(f)
	header := []string{"url", "start_time", "latency"}
	if err := writer.Write(header); err != nil {
		log.Fatal(err)
	}
	for _, entry := range s.entries {
		record := make([]string, 3)
		record[0] = entry.url
		record[1] = fmt.Sprintf("%f", float64(entry.startTime.Sub(t).Microseconds())/1e6)
		record[2] = fmt.Sprintf("%f", float64(entry.latency)/1e3)
		if err := writer.Write(record); err != nil {
			log.Fatal(err)
		}
	}
	writer.Flush()
	if err := f.Close(); err != nil {
		log.Fatal(err)
	}
}

type Request struct {
	client      *http.Client
	stats       *Stats
	contentType string
	imageBytes  []byte
	maxTrials   int
}

func sendRequest(wg *sync.WaitGroup, url string, request *Request) {
	defer wg.Done()
	stats := request.stats
	t := time.Now()
	trials := 0
	succeeded := false
	for !succeeded && trials < request.maxTrials {
		trials += 1
		resp, err := request.client.Post(
			url, request.contentType, bytes.NewReader(request.imageBytes))
		if err != nil {
			log.Printf("#trials: %d, err: %s\n", trials, err)
			continue
		}
		succeeded = true
		_, err = io.ReadAll(resp.Body)
		if err != nil {
			log.Println(err)
		}
		resp.Body.Close()
		duration := time.Since(t)
		if resp.StatusCode == 200 {
			stats.Succeeded(t, duration.Microseconds(), url)
		} else {
			stats.Failed(t, duration.Microseconds(), url)
			log.Printf("response code=%d\n", resp.StatusCode)
		}
	}
	if !succeeded {
		stats.Failed(t, time.Since(t).Microseconds(), url)
	}
}

type IntervalGenerator interface {
	Generate(count int) []time.Duration
}

type HalfUniformGenerator struct {
	IntervalGenerator
	unitTime int
}

func (g *HalfUniformGenerator) Generate(count int) []time.Duration {
	interval := 500 * time.Millisecond / time.Duration(count) * time.Duration(g.unitTime)
	var intervals []time.Duration
	for i := 0; i < count; i++ {
		intervals = append(intervals, interval)
	}
	return intervals
}

type PoissonGenerator struct {
	IntervalGenerator
	source   xrand.Source
	unitTime int
}

func (g *PoissonGenerator) Generate(count int) []time.Duration {
	lambda := 1000.0 / float64(count) * float64(g.unitTime)
	poisson := distuv.Poisson{
		Lambda: lambda,
		Src:    g.source,
	}
	var intervals []time.Duration
	for i := 0; i < count; i++ {
		interval := time.Millisecond * time.Duration(poisson.Rand())
		intervals = append(intervals, interval)
	}
	return intervals
}

func produceUrl(wg *sync.WaitGroup, url string, count int, request *Request, generator IntervalGenerator) {
	defer wg.Done()
	if count == 0 {
		return
	}
	requestWg := &sync.WaitGroup{}
	// t := time.Now()
	for _, interval := range generator.Generate(count) {
		requestWg.Add(1)
		go sendRequest(requestWg, url, request)
		// log.Println(time.Since(t))
		time.Sleep(interval)
	}
	requestWg.Wait()
}

func Process(file *os.File, request *Request, generator IntervalGenerator, unitTime int) {
	scanner := bufio.NewScanner(file)
	type Entry map[string]interface{}
	var entries []Entry

	for scanner.Scan() {
		var entry map[string]interface{}
		if err := json.Unmarshal([]byte(scanner.Text()), &entry); err != nil {
			log.Fatal(err)
		}
		entries = append(entries, entry)
	}
	t := time.Now()
	wg := &sync.WaitGroup{}
	for _, entry := range entries {
		ts := int(entry["ts"].(float64))
		if ts > 0 {
			timeToSleep := time.Duration(ts*unitTime)*time.Second - time.Since(t)
			if timeToSleep < 0 {
				log.Printf("time is off. too little number of workers. timeToSleep: %s\n", timeToSleep)
			} else {
				time.Sleep(timeToSleep)
			}
		}
		// log.Printf("start ts=%d at %s\n", ts, time.Since(t))
		for url, count := range entry["urls"].(map[string]interface{}) {
			wg.Add(1)
			go produceUrl(wg, url, int(count.(float64)), request, generator)
		}
	}
	wg.Wait()
}

func readImage(path string) []byte {
	f, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	bytes, err := io.ReadAll(f)
	if err != nil {
		log.Fatal(err)
	}
	return bytes
}

func doWork(client *http.Client, ctx context.Context, wg *sync.WaitGroup, stats *Stats, url string, contentType string, imageBytes *[]byte) {
	defer wg.Done()
	for {
		select {
		case <-ctx.Done():
			return
		default:
			t := time.Now()
			resp, err := client.Post(url, contentType, bytes.NewReader(*imageBytes))
			if err != nil {
				log.Println(err)
				stats.Failed(t, time.Since(t).Microseconds(), url)
				continue
			}
			_, err = io.ReadAll(resp.Body)
			if err != nil {
				log.Println(err)
			}
			resp.Body.Close()
			duration := time.Since(t)
			if resp.StatusCode == 200 {
				stats.Succeeded(t, duration.Microseconds(), url)
			} else {
				stats.Failed(t, duration.Microseconds(), url)
				log.Printf("response code=%d\n", resp.StatusCode)
			}
		}
	}
}

func main() {
	inputPath := flag.String("i", "", "input file path")
	numWorkers := flag.Int("n", 0, "num concurrent workers")
	startTimeOutPath := flag.String("start_time_out", "", "start time output path")
	latencyOutPath := flag.String("out", "", "latency output path")
	imagePath := flag.String("img", "", "image path")
	url := flag.String("url", "", "url for closed loop test")
	duration := flag.Int("d", 10, "duration in seconds")
	statsOutPath := flag.String("stats_out", "", "stat output path")
	maxIdleConn := flag.Int("max_idle_conn", 2, "max idle conn per host")
	intervalType := flag.String("interval_type", "halfuniform", "interval generator type")
	seed := flag.Int64("seed", -1, "random seed")
	maxTrials := flag.Int("max_trials", 1, "max trials for each request")
	unitTime := flag.Int("unit_time", 1, "unit time for count")
	flag.Parse()
	if len(*inputPath) == 0 && len(*url) == 0 {
		log.Fatal("-i or -url is required")
	}
	if len(*imagePath) == 0 {
		log.Fatal("-img is required")
	}

	imageBytes := readImage(*imagePath)
	contentType := http.DetectContentType(imageBytes)
	log.Printf("content type: %s\n", contentType)

	stats := &Stats{
		mutex:   &sync.Mutex{},
		entries: make([]StatEntry, 0),
	}

	client := &http.Client{Transport: &http.Transport{MaxIdleConnsPerHost: *maxIdleConn}}

	t := time.Now()
	if len(*inputPath) > 0 {
		file, err := os.Open(*inputPath)
		if err != nil {
			log.Fatal(err)
		}
		request := &Request{
			client:      client,
			stats:       stats,
			contentType: contentType,
			imageBytes:  imageBytes,
			maxTrials:   *maxTrials,
		}
		var generator IntervalGenerator
		switch *intervalType {
		case "halfuniform":
			generator = &HalfUniformGenerator{}
		case "poisson":
			var srcSeed uint64
			if *seed == -1 {
				srcSeed = uint64(time.Now().UnixNano())
			} else {
				srcSeed = uint64(*seed)
			}
			log.Printf("poisson seed: %d", srcSeed)
			generator = &PoissonGenerator{
				source:   xrand.NewSource(srcSeed),
				unitTime: *unitTime,
			}
		default:
			log.Fatalf("unknown interval type=%s\n", *intervalType)
		}
		Process(file, request, generator, *unitTime)
	} else {
		if *numWorkers <= 0 {
			log.Fatal("-n (> 0) is required")
		}

		log.Printf("num workers: %d\n", *numWorkers)

		// closed loop test
		ctx, cancel := context.WithCancel(context.Background())

		wg := &sync.WaitGroup{}
		wg.Add(*numWorkers)
		for i := 0; i < *numWorkers; i++ {
			go doWork(client, ctx, wg, stats, *url, contentType, &imageBytes)
		}
		time.Sleep(time.Duration(*duration) * time.Second)
		cancel()
		wg.Wait()
	}

	log.Printf("elapsed time: %s\n", time.Since(t))
	log.Printf("mean: %f ms\n", stats.Mean())
	log.Printf("# success: %d\n", stats.success_count)
	log.Printf("# failure: %d\n", stats.failure_count)

	if len(*latencyOutPath) > 0 {
		stats.OutputLatencies(*latencyOutPath)
	}
	if len(*startTimeOutPath) > 0 {
		stats.OutputStartTimes(t, *startTimeOutPath)
	}
	if len(*statsOutPath) > 0 {
		stats.Output(t, *statsOutPath)
	}
}
