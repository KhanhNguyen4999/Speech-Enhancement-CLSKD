import torch

class DCCRN():
    def __init__(self, model):
        # Get input size from model
        self.model = model
        
        #feature maps contained encoder, decoder, LSTM 
        self.feature_maps = {"encoder":[],"decoder":[],"clstm":[]}
        # Register forward hooks for encoder, decoder, and enhance modules
        self.handle_encoder = [self.model.encoder[i].register_forward_hook(self.encoder_hook) for i in range(len(self.model.encoder))]
        self.handle_decoder = [self.model.decoder[i].register_forward_hook(self.decoder_hook) for i in range(len(self.model.decoder))]
        self.handle_clstm = self.model.enhance.register_forward_hook(self.enhance_hook)

    def encoder_hook(self, module, input, output):
        """
        Append encoder feature map to the list of feature maps
        """
        self.feature_maps["encoder"].append(output)
        

    def decoder_hook(self, module, input, output):
        """
        Append decoder feature map to the list of feature maps
        """
        self.feature_maps["decoder"].append(output)

    def enhance_hook(self, module, input, output):
        """
        Append enhance (Complex-LSTM) feature map to the list of feature maps
        """
        self.feature_maps["clstm"].append(output)

    def remove_hook(self):
        """"
        Remove hook register to prevent memory leaking
        """
        [self.handle_encoder[i].remove() for i in range(len(self.model.encoder))]
        [self.handle_decoder[i].remove() for i in range(len(self.model.decoder))]
        self.handle_clstm.remove()

    def extract_feature_maps(self, input):
        """
        Extract feature maps from model's encoder, decoder, and enhance modules
        """
     
        # Pass input through the model to extract feature maps
        with torch.no_grad():
            self.model(input)
    
        return self.feature_maps
    