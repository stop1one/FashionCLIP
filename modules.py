from torch import nn
import timm
from transformers import DistilBertModel, DistilBertConfig
import config as Config


class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    We'll use a ResNet50 as the image encoder.
    This encode each image to a fixed size vector with the size 2048. (in case of ResNet50)
    """
    def __init__(self,
    model_name=Config.model_name, 
    pretrained=Config.pretrained, 
    trainable=Config.trainable
    ):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained, num_classes=0, global_pool="avg")
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class TextEncoder(nn.Module):
    """
    We'll use DistilBERT as the text encoder.
    In the case of DistilBERT (and also BERT) the output hidden representation for each token is a vector with size 768.
    """
    def __init__( self,
        model_name=Config.text_encoder_model, 
        pretrained=Config.pretrained,
        trainable=Config.trainable
        ):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        """
        we are using the CLS token hidden representation as the sentence's embedding
        CLS -> first, SEP -> Seperator btw sentences
        """
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


class ProjectionHead(nn.Module):
    """
    Encode the images and texts into fixed size vectors (same dimension).
    Image-2048(ResNet50), Text-768(DistilBERT) => Output-256(Below)
    """
    def __init__(
        self,
        embedding_dim,
        projection_dim=Config.projection_dim,   # In config.py
        dropout=Config.dropout  # = 0.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

