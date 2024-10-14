python main_depth.py \
  --depth-encoder vitl \
  --output-path video_depth_vis \
  --depth-grayscale \
  --dataset-name ego4d_video \
  --anno-path /fs/fast/u2023000902/annotations/egovlpv3/EgoClip_hardnegHOI.csv \
  --depth-input-size 518 \
  --output-path outputs/test \
  --image-path /fs/fast/u2023000902/data/ego4d/down_scale \
  --clip-length 8
#   [--input-size <size>] [--pred-only] [--grayscale]

# python main_depth_origin.py \
#   --encoder vitl \
#   --video-path depth_anything_v2/assets/examples_video --outdir video_depth_vis \