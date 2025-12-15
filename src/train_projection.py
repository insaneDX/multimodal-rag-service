"""
Train an MLP projection head to map CLIP image embeddings to text embedding space.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

# Allow local 'src' imports
sys.path.append(str(Path(__file__).parent.parent))

import src.logger_config
from loguru import logger

# ----- Config -----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-4
SAVE_PATH = os.path.join(Path(__file__).parent.parent, "mlp_projection.pt")

# Models to use
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DATASET_NAME = "comet-team/coco-500"


def main():
    logger.info("Training projection head (Image → Text space)")
    logger.info("Device: {}", DEVICE)

    # ----- Load Encoders -----
    logger.info("Loading encoders...")

    from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
    from datasets import load_dataset

    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    text_encoder = AutoModel.from_pretrained(TEXT_MODEL_NAME).to(DEVICE)
    text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

    # Freeze encoders (we only train the projection MLP)
    for p in clip_model.parameters():
        p.requires_grad = False
    for p in text_encoder.parameters():
        p.requires_grad = False

    logger.info("Encoders loaded and frozen.")

    # ----- MLP Projection Layer -----
    class ProjectionMLP(nn.Module):
        """
        Simple MLP to project from CLIP's 512-dim space to text space.
        """
        def __init__(self, in_dim=512, out_dim=384, hidden=512):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, out_dim)
            )

        
        def forward(self, x):
            """
            Forward pass through the MLP.

            Args:
                x (torch.Tensor): Input tensor with shape (batch_size, 512)

            Returns:
                torch.Tensor: Output tensor with shape (batch_size, 384)
    
            Input tensor is passed through the MLP, consisting of 2 linear
            layers with ReLU activation in between.

            Output tensor has the same batch size as the input, but with
            384 features instead of 512.
            """
            return self.net(x)

    projection = ProjectionMLP(in_dim=512, out_dim=384).to(DEVICE)
    optimizer = optim.Adam(projection.parameters(), lr=LR)
    criterion = nn.MSELoss()

    logger.info("Projection model initialized.")

    # ----- Dataset Prep -----
    logger.info("Loading {} dataset for training...", DATASET_NAME)
    try:
        dataset = load_dataset(DATASET_NAME, split="train")
    except Exception as e:
        logger.error("Failed to load dataset: {}", e)
        sys.exit(1)

    logger.info("Dataset loaded with {} samples and columns {}", 
                len(dataset), dataset.column_names)

    # ----- Collate Function -----
    def collate_fn(batch):
        """
        Custom collate function that:
        1. Filters out invalid samples
        2. Converts images to RGB
        3. Creates captions from category fields
        4. Computes embeddings for both modalities

        Args:
            batch (list of dict): A list of dictionaries containing the batch data.
                Each item should contain the following keys: "Image", "Category 5", "Category 10".

        Returns:
            img_embeds (torch.Tensor): Image embeddings with shape (batch_size, 512)
            txt_embeds (torch.Tensor): Text embeddings with shape (batch_size, 384)
        """
        # Clean entries: keep items with a valid image
        batch = [item for item in batch if item.get("Image") is not None]
        if not batch:
            return None, None

        # Convert PIL images to RGB mode
        images = [item["Image"].convert("RGB") for item in batch]

        # Build descriptive captions from category fields
        captions = []
        for item in batch:
            categories = []
            if item.get("Category 5"):
                categories.append(str(item["Category 5"]))
            if item.get("Category 10"):
                categories.append(str(item["Category 10"]))
            caption = (
                f"An image containing the objects: {', '.join(categories)}."
                if categories else "A general image."
            )
            captions.append(caption)

        # Compute image embeddings (frozen CLIP encoder)
        img_inputs = clip_processor(
            images=images, 
            return_tensors="pt", 
            padding=True
        ).to(DEVICE)
        
        with torch.no_grad():
            # No gradients needed for frozen model
            img_embeds = clip_model.get_image_features(**img_inputs)

        # Compute text embeddings (frozen text encoder)
        txt_inputs = text_tokenizer(
            captions, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(DEVICE)
        
        with torch.no_grad():
            # No gradients needed for frozen model
            txt_hidden = text_encoder(**txt_inputs).last_hidden_state
            txt_embeds = txt_hidden.mean(dim=1)

        return img_embeds, txt_embeds

    # Filter dataset entries with missing 'Image'
    fixed_dataset = dataset.filter(lambda e: e["Image"] is not None)
    dataloader = DataLoader(
        fixed_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn
    )

    logger.info("Dataset ready with {} usable samples", len(fixed_dataset))

    # ----- Training Loop -----
    logger.info("Starting training loop...")
    
    for epoch in range(EPOCHS):
        projection.train()
        total_loss = 0.0
        count = 0

        for step, batch in enumerate(dataloader):
            if batch is None or batch[0] is None:
                continue

            img_embeds, txt_embeds = batch
            pred = projection(img_embeds)

            loss = criterion(pred, txt_embeds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

            if (step + 1) % 10 == 0:
                logger.info(
                    "Epoch [{}/{}], Step {}, Loss: {:.4f}",
                    epoch + 1, EPOCHS, step + 1, loss.item()
                )

        avg_loss = total_loss / max(count, 1)
        logger.info("Epoch {} completed — Avg Loss: {:.4f}", epoch + 1, avg_loss)

    # ----- Save Model -----
    logger.info("Saving projection model to '{}'", SAVE_PATH)
    torch.save(projection.state_dict(), SAVE_PATH)

    logger.success("Training complete. Model saved and ready for inference!")


if __name__ == "__main__":
    main()