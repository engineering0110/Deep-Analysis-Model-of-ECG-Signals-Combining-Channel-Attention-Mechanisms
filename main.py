class MeanPoolingMIL(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.embedding = nn.Linear(cfgs['feature_dim'], cfgs['feature_dim'])
        self.dropout = nn.Dropout(.2)
        self.ln = nn.LayerNorm(cfgs['feature_dim'])
        self.classifier = nn.Linear(cfgs['feature_dim'], cfgs["num_classes"])
        self.cfg = cfgs
        
    def forward(self, x):
        x = self.dropout(self.embedding(x))
        x = self.ln(x)
        y = self.classifier(x)
        # y = F.sigmoid(y)
        return y, x
    
    def inference(self, x):
        atten_target, atten_score = self.attentionnet(x)
        x = x.reshape([-1, self.cfg['feature_dim']])
        y, _ = self.forward(x)
        return torch.softmax(y, 1), atten_target, atten_score[0]
