import torch

class DCCRN():
    def __init__(self, model):
        # Get input size from model
        self.model = model
        
        #feature maps contained encoder, decoder, LSTM 
        self.feature_maps = []

        # Register forward hooks for encoder, decoder, and enhance modules
        self.handle_encoder = self.model.encoder[-1].register_forward_hook(self.encoder_hook)
        self.handle_decoder = self.model.decoder[-1].register_forward_hook(self.decoder_hook)
        self.handle_clstm = self.model.enhance.register_forward_hook(self.enhance_hook)

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

    def remove_hook(self):
        """"
        Remove hook register to prevent memory leaking
        """
        self.handle_encoder.remove()
        self.handle_decoder.remove()
        self.handle_clstm.remove()

    def extract_feature_maps(self, input):
        """
        Extract feature maps from model's encoder, decoder, and enhance modules
        """
        # Reset the list of feature maps
        self.feature_maps = []
     
        # Pass input through the model to extract feature maps
        with torch.no_grad():
            self.model(input)
    
        return self.feature_maps
    