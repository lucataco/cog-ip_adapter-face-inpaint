# Prediction interface for Cog
from cog import BasePredictor, Input, Path
import os
import sys
import cv2
sys.path.extend(['/IP-Adapter'])
os.system("rm -rf /IP-Adapter/models/image_encoder")
os.system("ln -s /src/image_encoder /IP-Adapter/models/image_encoder")
import torch
import tempfile
import mimetypes
import numpy as np
from PIL import Image
import mediapipe as mp
from typing import List
from PIL import Image, ImageFilter
from diffusers import StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from ip_adapter import IPAdapter

base_model_path = "runwayml/stable-diffusion-v1-5"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "/IP-Adapter/models/image_encoder/"
ip_ckpt = "/IP-Adapter/models/ip-adapter_sd15.bin"
device = "cuda"
MODEL_CACHE = "model-cache"
VAE_CACHE = "vae-cache"

def face_mask_google_mediapipe(
    images: List[Image.Image], blur_amount: float = 0.0, bias: float = 50.0
) -> List[Image.Image]:
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh

    face_detection = mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.1
    )
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, min_detection_confidence=0.1
    )
    masks = []
    for image in images:
        image_np = np.array(image)
        # Perform face detection
        results_detection = face_detection.process(image_np)
        ih, iw, _ = image_np.shape
        if results_detection.detections:
            this_im_masks = []
            for detection in results_detection.detections:
                bboxC = detection.location_data.relative_bounding_box
                bbox = (
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih),
                )
                # make sure bbox is within image
                bbox = (
                    max(0, bbox[0]),
                    max(0, bbox[1]),
                    min(iw - bbox[0], bbox[2]),
                    min(ih - bbox[1], bbox[3]),
                )
                print(bbox)
                # Extract face landmarks
                face_landmarks = face_mesh.process(
                    image_np[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
                ).multi_face_landmarks
                indexes = [
                    10,
                    338,
                    297,
                    332,
                    284,
                    251,
                    389,
                    356,
                    454,
                    323,
                    361,
                    288,
                    397,
                    365,
                    379,
                    378,
                    400,
                    377,
                    152,
                    148,
                    176,
                    149,
                    150,
                    136,
                    172,
                    58,
                    132,
                    93,
                    234,
                    127,
                    162,
                    21,
                    54,
                    103,
                    67,
                    109,
                ]
                if face_landmarks:
                    mask = Image.new("L", (iw, ih), 0)
                    mask_np = np.array(mask)
                    for face_landmark in face_landmarks:
                        face_landmark = [face_landmark.landmark[idx] for idx in indexes]
                        landmark_points = [
                            (int(l.x * bbox[2]) + bbox[0], int(l.y * bbox[3]) + bbox[1])
                            for l in face_landmark
                        ]
                        mask_np = cv2.fillPoly(
                            mask_np, [np.array(landmark_points)], 255
                        )
                    mask = Image.fromarray(mask_np)
                    # Apply blur to the mask
                    if blur_amount > 0:
                        mask = mask.filter(ImageFilter.GaussianBlur(blur_amount))
                    # Apply bias to the mask
                    if bias > 0:
                        mask = np.array(mask)
                        mask = mask + bias * np.ones(mask.shape, dtype=mask.dtype)
                        mask = np.clip(mask, 0, 255)
                        mask = Image.fromarray(mask)
                    # Convert mask to 'L' mode (grayscale) before saving
                    mask = mask.convert("L")
                    this_im_masks.append(mask)
                else:
                    # If face landmarks are not available, add a black mask of the same size as the image
                    this_im_masks.append(Image.new("L", (iw, ih), 255))
            masks.append(this_im_masks)
        else:
            print("No face detected, adding full mask")
            # If no face is detected, add a white mask of the same size as the image
            masks.append([Image.new("L", (iw, ih), 255)])
    return masks

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        vae = AutoencoderKL.from_pretrained(
            vae_model_path,
            cache_dir=VAE_CACHE
        ).to(dtype=torch.float16)
        # load SD pipeline
        self.pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None,
            cache_dir=MODEL_CACHE
        )

    @torch.inference_mode()
    def predict(
        self,
        face_image: Path = Input(
             description="Input face image",
        ),
        blur_amount: float = Input(
            description="Blur to apply to mask to face", default=0.0
        ),
        source_image: Path = Input(
             description="Source image of body",
             default=None
        ),
        strength: float = Input(
             description="mask strength",
             default=0.7
        ),
        prompt: str = Input(
            description="Prompt",
            default=""
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        # constants
        num_inference_steps = 50

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        face_image = Image.open(face_image)
        face_image.resize((256, 256))

        # Create mask from source image
        tmp_out_dir = tempfile.mkdtemp()
        mt = mimetypes.guess_type(str(source_image))
        if mt and mt[0] and mt[0].startswith("image/"):
            image = [Image.open(str(source_image)).convert("RGB")]
        seg_masks = face_mask_google_mediapipe(
            images=image, blur_amount=blur_amount, bias=0
        )
        mask_paths = []
        for i, mask_list in enumerate(seg_masks):
            for j, mask in enumerate(mask_list):
                mask_file = f"{i}_{j}.mask.png"
                mask_path = os.path.join(tmp_out_dir, mask_file)
                mask_paths.append(mask_path)
                mask.save(mask_path)
        
        # Continue with face-inpaint
        source_image = Image.open(source_image).convert("RGB")
        source_image.resize((512, 512*source_image.height//source_image.width))

        mask = Image.open(mask_paths[0])
        mask.resize(mask.size)

        ip_model = IPAdapter(self.pipe, image_encoder_path, ip_ckpt, device)

        images = ip_model.generate(
            pil_image=face_image,
            image=source_image,
            mask_image=mask,
            strength=strength,
            num_samples=num_outputs,
            num_inference_steps=num_inference_steps,
            seed=seed,
            prompt=prompt
        )

        output_paths = []
        for i, _ in enumerate(images):
            output_path = f"/tmp/out-{i}.png"
            images[i].save(output_path)
            output_paths.append(Path(output_path))
            
        return output_paths
