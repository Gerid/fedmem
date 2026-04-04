FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# System deps for matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Python deps only — code lives on Network Volume
RUN pip install --no-cache-dir \
    numpy>=1.24 \
    "river>=0.23" \
    "scikit-learn>=1.3" \
    "scipy>=1.10" \
    "matplotlib>=3.7" \
    "pyyaml>=6.0" \
    "torchvision>=0.16" \
    runpod

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
