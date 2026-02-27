from datetime import datetime

from lightx2v.pipeline import LightX2VPipeline


def main():
    ts = datetime.now().strftime("%y%m%d%H%M")

    pipe = LightX2VPipeline(
        model_path="/path/to/JunhaoZhuang/FlashVSR-v1.1",
        model_cls="flashvsr",
        task="sr",
    )

    pipe.create_generator(
        config_json="/path/to/LightX2V/configs/flashvsr/flashvsr_v11_tiny_sr.json"
    )

    seed = 42
    prompt = ""
    negative_prompt = ""

    input_video_path = "/path/to/input_video.mp4"

    save_result_path = f"/path/to/output/flashvsr_sr_{ts}.mp4"

    pipe.generate(
        seed=seed,
        prompt=prompt,
        negative_prompt=negative_prompt,
        save_result_path=save_result_path,
        video_path=input_video_path,
        sr_ratio=1.5,
    )


if __name__ == "__main__":
    main()
