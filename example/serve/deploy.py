import argparse
import io
import time

import torch
import torchvision
import torchvision.transforms as transforms

import ray
from ray import serve
from PIL import Image


def main(args):
    ray.init(address=args.address, namespace="serve")
    serve.start(detached=True, dedicated_cpu=args.serve_dedicated_cpu,
                http_options={"host": "0.0.0.0"})

    autoscale_config = None
    if args.autoscale:
        autoscale_config = {}
        if args.autoscale_target_num:
            autoscale_config["target_num_ongoing_requests_per_replica"] = \
                args.autoscale_target_num
        if args.autoscale_min_replicas:
            autoscale_config["min_replicas"] = args.autoscale_min_replicas
        if args.autoscale_max_replicas:
            autoscale_config["max_replicas"] = args.autoscale_max_replicas
        if args.autoscale_upscale_delay:
            autoscale_config["upscale_delay_s"] = args.autoscale_upscale_delay
        if args.autoscale_downscale_delay:
            autoscale_config["downscale_delay_s"] = \
                args.autoscale_downscale_delay
        if args.autoscale_metrics_interval:
            autoscale_config["metrics_interval_s"] = \
                args.autoscale_metrics_interval
        if args.autoscale_look_back_period:
            autoscale_config["look_back_period_s"] = \
                args.autoscale_look_back_period
        print("Autoscaling enabled. kwargs=%s" % autoscale_config)

    @serve.deployment(
        name=args.name,
        num_replicas=args.num_replicas,
        ray_actor_options={"num_cpus": args.num_cpus},
        route_prefix="/" + args.name,
        max_concurrent_queries=args.max_concurrent_queries,
        autoscaling_config=autoscale_config,
        version="v1")
    class Classifier:
        def __init__(self):
            func = getattr(torchvision.models, args.model)
            self.model = func(pretrained=False).eval()
            self.preprocessor = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # remove alpha channel
                transforms.Lambda(lambda t: t[:3, ...]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

        async def __call__(self, starlette_request):
            img_bytes = await starlette_request.body()
            pil_img = Image.open(io.BytesIO(img_bytes))
            input_tensor = self.preprocessor(pil_img).unsqueeze(0)
            with torch.no_grad():
                output_tensor = self.model(input_tensor)
            return {"class_index": int(torch.argmax(output_tensor[0]))}

    Classifier.deploy()

    if args.address == "local":
        while True:
            time.sleep(100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="auto")
    parser.add_argument("--num-replicas", type=int)
    parser.add_argument("--num-cpus", type=float, default=1)
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--name", default="classifier")
    parser.add_argument("--max-concurrent-queries", default=1000, type=int)
    parser.add_argument("--autoscale", action="store_true",
                        help="enable autoscaling")
    parser.add_argument("--autoscale-target-num", type=int,
                        help="autoscaling target num ongoing requests")
    parser.add_argument("--autoscale-min-replicas", type=int, default=1)
    parser.add_argument("--autoscale-max-replicas", type=int, default=10)
    parser.add_argument("--autoscale-upscale-delay", type=float)
    parser.add_argument("--autoscale-downscale-delay", type=float)
    parser.add_argument("--autoscale-metrics-interval", type=float)
    parser.add_argument("--autoscale-look-back-period", type=float)
    parser.add_argument("--serve-dedicated-cpu", action="store_true")

    main(parser.parse_args())
