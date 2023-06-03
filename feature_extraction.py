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
    

class DCCRNet():
    def __init__(self, model):
        # Get input size from model
        self.model = model
        
        #feature maps contained encoder, decoder, LSTM 
        self.clstm = []
        self.encoder = []
        self.decoder = []
        self.feature_maps = {"encoder":[],"decoder":[],"clstm_real":[],"clstm_img":[]}
        # Register forward hooks for encoder, decoder, and enhance modules
        self.handler_encoder = [model.masker.encoders[i].register_forward_hook(self.encoder_hook) for i in range(len(self.model.masker.encoders)-1)]
        self.handler_decoder = [model.masker.decoders[i].register_forward_hook(self.decoder_hook) for i in range(len(self.model.masker.decoders))]
        self.handler_lstm = model.masker.encoders[-1].rnn.rnns[1].register_forward_hook(self.lstm_hook)
        

    def encoder_hook(self, module, input, output):
        """
        Append encoder feature map to the list of feature maps
        """
        self.encoder.append(output)
        

    def decoder_hook(self, module, input, output):
        """
        Append decoder feature map to the list of feature maps
        """
        self.decoder.append(output)

    def lstm_hook(self, module, input, output):
        """
        Append Complex-LSTM feature map to the list of feature maps
        """
        self.clstm.append(output)
    
    def remove_hook(self):
        """"
        Remove hook register to prevent memory leaking
        """
        [self.handler_encoder[i].remove() for i in range(len(self.model.masker.encoders)-1)]
        [self.handler_decoder[i].remove() for i in range(len(self.model.masker.decoders))]
        self.handler_lstm.remove()

    def extract_feature_maps(self, input):
        """
        Extract feature maps from model's encoder, decoder, and enhance modules
        """
     
        # Pass input through the model to extract feature maps
        with torch.no_grad():
            self.model(input)
    
        # Handle with complex LSTM - complex number
        self.feature_maps["clstm_real"].append(self.clstm[0].real)
        self.feature_maps["clstm_img"].append(self.clstm[0].imag)

        # Handle with encoders
        for i in range(len(self.encoder)):
            f_real = self.encoder[i].real
            f_img = self.encoder[i].imag
            self.feature_maps["encoder"].append(torch.cat([f_real,f_img],dim=1))
        
        #Handle with decoders
        for i in range(len(self.decoder)):
            f_real = self.decoder[i].real
            f_img = self.decoder[i].imag
            self.feature_maps["decoder"].append(torch.cat([f_real,f_img],dim=1))

        return self.feature_maps
    
    