# Placeholder — sample images and model files are downloaded at runtime
# by src/01_local_inference.py
#
# After running the workshop, this directory will contain:
#   models/
#   ├── sample_image.jpg         ← Downloaded sample image
#   ├── imagenet_classes.txt     ← ImageNet label mapping
#   ├── exported/
#   │   ├── mobilenet_v2.onnx    ← ONNX exported model
#   │   └── mobilenet_v2_exported.pt2   ← torch.export model
#   └── compiled/
#       ├── mobilenet_v2_compiled.tflite  ← AI Hub compiled model
#       └── mobilenet_v2_int8.tflite      ← INT8 quantized model
