FROM golang:1.18

ADD go.mod /go/src
ADD go.sum /go/src
ADD main.go /go/src
ADD image.jpg /go
RUN cd /go/src && go build && mv loadgen /go
RUN apt-get update && apt-get install -y \
    tmux \
    && rm -rf /var/lib/apt/lists/*
