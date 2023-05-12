import torch

class DCCRN():
    def __init__(self, model):
        # Get input size from model
        self.model = model
        #self.input_size = self.model.stft.weight.size(1)
        #self.input = torch.randn(batch_szie, self.input_size)
        
        #feature maps contained encoder, decoder, LSTM 
        self.feature_maps = []

        # Register forward hooks for encoder, decoder, and enhance modules
        self.model.encoder[-1].register_forward_hook(self.encoder_hook)
        self.model.decoder[-1].register_forward_hook(self.decoder_hook)
        self.model.enhance.register_forward_hook(self.enhance_hook)

    def encoder_hook(self, module, input, output):
        """
        Append encoder feature map to the list of feature maps
        """
        self.feature_maps.append(output)

    def decoder_hook(self, module, input, output):
        """
        Append decoder feature map to the list of feature maps
        """
        self.feature_maps.append(output)

    def enhance_hook(self, module, input, output):
        """
        Append enhance (LSTM) feature map to the list of feature maps
        """
        self.feature_maps.append(output)

    def extract_feature_maps(self, input):
        """
        Extract feature maps from model's encoder, decoder, and enhance modules
        """
        # Reset the list of feature maps
        self.feature_maps = []
     
        # Pass input through the model to extract feature maps
        with torch.no_grad():
            self.model(input)
        
        # if features == 'encoder':
        #     self.feature_maps = self.feature_maps[0]
        # elif features == 'decoder':
        #     self.feature_mapss = self.feature_maps[2]
        # elif features == 'CLSTM':
        #     self.feature_maps = self.feature_maps[1]

        return self.feature_maps
    