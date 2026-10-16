import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm



class EncoderProjectorConcat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.encoder_projector_ds_rate # Es 5
        self.encoder_dim = config.encoder_dim      # Es 768
        self.llm_dim = config.llm_dim              # Es 2048
        #print("k: ",self.k)
        #print("enconder dim:", self.encoder_dim)
        #print("llm dim:",self.llm_dim)
        
        # 768 * 5 = 3840. Esto es lo que mat1 y mat2 necesitan.
        input_dim = self.encoder_dim * self.k 
        
        self.linear1 = nn.Linear(input_dim, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, self.llm_dim)

        #Nuevo
        self.hidden_activations = None 
      

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        
        # Ajuste de frames
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        
        # Re-shape (Concatenación)
        # Esto pasa de [4, 1500, 768] a [4, 300, 3840]
        x = x.reshape(batch_size, -1, dim * self.k)

        #print(f"DEBUG - Dimensión después de concatenar: {x.shape[-1]}")
        #print(f"DEBUG - Dimensión que espera linear1: {self.linear1.in_features}")

        x = self.linear1(x)
        x = self.relu(x)
        #NUEVO

        self.hidden_activations = x
        if x.requires_grad:
           self.hidden_activations.retain_grad()
        #self.hidden_activations.retain_grad()
        x = self.linear2(x)

        return x


class EncoderProjectorCov1d(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        self.encoder_dim = 768  # Forzamos la realidad de Whisper
        self.llm_dim = 2048      # Dimensión de TinyLlama

        # Si llega algo de 2048, lo bajaremos a 768 primero con esta capa de seguridad
        self.input_layer = nn.Linear(2048, 768) 
        
        self.conv1d = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=self.k, stride=self.k)
        self.linear1 = nn.Linear(768, 2048)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(2048, self.llm_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        # x shape: [Batch, Seq, Channels]
        
        # Si llega con 2048, lo proyectamos a 768 antes de la Conv1d
        if x.shape[-1] == 2048:
            x = self.input_layer(x)
        
        # Ahora x siempre es [Batch, Seq, 768]
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = x.transpose(1, 2)
        
        x = self.relu1(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        return x


class EncoderProjectorQFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        from transformers import Blip2QFormerConfig, Blip2QFormerModel
        configuration = Blip2QFormerConfig()
        configuration.encoder_hidden_size = self.encoder_dim
        configuration.num_hidden_layers = config.qformer_layers

        self.query_len = int(config.get("query_len", 64))
        self.query = nn.Parameter(torch.zeros(1, self.query_len, configuration.hidden_size))
        self.query.data.normal_(mean=0.0, std=1.0)
        self.qformer = Blip2QFormerModel(configuration)

        self.linear = nn.Linear(configuration.hidden_size, self.llm_dim)
        self.norm = nn.LayerNorm(self.llm_dim, eps=1e-5)

    def forward(self, x, atts):
        query = self.query.expand(x.shape[0], -1, -1)
        
        query_output = self.qformer(
            query_embeds=query,
            encoder_hidden_states=x,
            encoder_attention_mask=atts,
            return_dict=True,
        )
        
        query_proj = self.norm(self.linear(query_output.last_hidden_state))
        
        return query_proj
