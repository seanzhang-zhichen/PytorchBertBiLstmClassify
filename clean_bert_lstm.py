import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer,BertModel
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

np.random.seed(0)
torch.manual_seed(0)
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    torch.cuda.manual_seed(0)


#剔除标点符号,\xa0 空格
def pretreatment(comments):
    result_comments=[]
    punctuation='。，？！：%&~（）、；“”&|,.?!:%&~();""'
    for comment in comments:
        comment= ''.join([c for c in comment if c not in punctuation])
        comment= ''.join(comment.split())   #\xa0
        result_comments.append(comment)
    
    return result_comments

class bert_lstm(nn.Module):
    def __init__(self, bertpath, hidden_dim, output_size,n_layers,bidirectional=True, drop_prob=0.5):
        super(bert_lstm, self).__init__()
 
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
        #Bert ----------------重点，bert模型需要嵌入到自定义模型里面
        self.bert=BertModel.from_pretrained(bertpath)
        for param in self.bert.parameters():
            param.requires_grad = True
        
        # LSTM layers
        self.lstm = nn.LSTM(768, hidden_dim, n_layers, batch_first=True,bidirectional=bidirectional)
        
        # dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        # linear and sigmoid layers
        if bidirectional:
            self.fc = nn.Linear(hidden_dim*2, output_size)
        else:
            self.fc = nn.Linear(hidden_dim, output_size)
          
        #self.sig = nn.Sigmoid()
 
    def forward(self, x, hidden):
        batch_size = x.size(0)
        #生成bert字向量
        x=self.bert(x)[0]     #bert 字向量
        
        # lstm_out
        #x = x.float()
        lstm_out, (hidden_last,cn_last) = self.lstm(x, hidden)
        #print(lstm_out.shape)   #[32,100,768]
        #print(hidden_last.shape)   #[4, 32, 384]
        #print(cn_last.shape)    #[4, 32, 384]
        
        #修改 双向的需要单独处理
        if self.bidirectional:
            #正向最后一层，最后一个时刻
            hidden_last_L=hidden_last[-2]
            #print(hidden_last_L.shape)  #[32, 384]
            #反向最后一层，最后一个时刻
            hidden_last_R=hidden_last[-1]
            #print(hidden_last_R.shape)   #[32, 384]
            #进行拼接
            hidden_last_out=torch.cat([hidden_last_L,hidden_last_R],dim=-1)
            #print(hidden_last_out.shape,'hidden_last_out')   #[32, 768]
        else:
            hidden_last_out=hidden_last[-1]   #[32, 384]
            
            
        # dropout and fully-connected layer
        out = self.dropout(hidden_last_out)
        #print(out.shape)    #[32,768]
        out = self.fc(out)
        
        return out
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        
        number = 1
        if self.bidirectional:
            number = 2
        
        if (USE_CUDA):
            hidden = (weight.new(self.n_layers*number, batch_size, self.hidden_dim).zero_().float().cuda(),
                      weight.new(self.n_layers*number, batch_size, self.hidden_dim).zero_().float().cuda()
                     )
        else:
            hidden = (weight.new(self.n_layers*number, batch_size, self.hidden_dim).zero_().float(),
                      weight.new(self.n_layers*number, batch_size, self.hidden_dim).zero_().float()
                     )
        
        return hidden

class ModelConfig:
    batch_size = 32
    output_size = 2
    hidden_dim = 384   #768/2
    n_layers = 2
    lr = 2e-5
    bidirectional = True  #这里为True，为双向LSTM
    # training params
    epochs = 10
    # batch_size=50
    print_every = 10
    clip=5 # gradient clipping
    use_cuda = USE_CUDA
    bert_path = 'bert-base-chinese' #预训练bert路径
    save_path = 'bert_bilstm.pth' #模型保存路径


def train_model(config, data_train):
    net = bert_lstm(config.bert_path, 
                    config.hidden_dim, 
                    config.output_size,
                    config.n_layers, 
                    config.bidirectional)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    if(config.use_cuda):
        net.cuda()
    net.train()
    for e in range(config.epochs):
        # initialize hidden state
        h = net.init_hidden(config.batch_size)
        counter = 0
        # batch loop
        for inputs, labels in data_train:
            counter += 1
            
            if(config.use_cuda):
                inputs, labels = inputs.cuda(), labels.cuda()
            h = tuple([each.data for each in h])
            net.zero_grad()
            output= net(inputs, h)
            loss = criterion(output.squeeze(), labels.long())
            loss.backward()
            optimizer.step()
    
            # loss stats
            if counter % config.print_every == 0:
                net.eval()
                with torch.no_grad():
                    val_h = net.init_hidden(config.batch_size)
                    val_losses = []
                    for inputs, labels in valid_loader:
                        val_h = tuple([each.data for each in val_h])

                        if(config.use_cuda):
                            inputs, labels = inputs.cuda(), labels.cuda()

                        output = net(inputs, val_h)
                        val_loss = criterion(output.squeeze(), labels.long())

                        val_losses.append(val_loss.item())
                net.train()
                print("Epoch: {}/{}, ".format(e+1, config.epochs),
                    "Step: {}, ".format(counter),
                    "Loss: {:.6f}, ".format(loss.item()),
                    "Val Loss: {:.6f}".format(np.mean(val_losses)))
    torch.save(net.state_dict(), config.save_path)                
    
def test_model(config, data_test):
    net = bert_lstm(config.bert_path, 
                config.hidden_dim, 
                config.output_size,
                config.n_layers, 
                config.bidirectional)
    net.load_state_dict(torch.load(config.save_path))
    net.cuda()
    criterion = nn.CrossEntropyLoss()
    test_losses = [] # track loss
    num_correct = 0
    
    # init hidden state
    h = net.init_hidden(config.batch_size)
    
    net.eval()
    # iterate over test data
    for inputs, labels in data_test:
        h = tuple([each.data for each in h])
        if(USE_CUDA):
            inputs, labels = inputs.cuda(), labels.cuda()
        output = net(inputs, h)
        test_loss = criterion(output.squeeze(), labels.long())
        test_losses.append(test_loss.item())
        
        output=torch.nn.Softmax(dim=1)(output)
        pred=torch.max(output, 1)[1]
        
        # compare predictions to true label
        correct_tensor = pred.eq(labels.long().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not USE_CUDA else np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)
    print("Test loss: {:.3f}".format(np.mean(test_losses)))
    # accuracy over all test data
    test_acc = num_correct/len(data_test.dataset)
    print("Test accuracy: {:.3f}".format(test_acc))

def predict(test_comment_list, config):
    net = bert_lstm(config.bert_path, 
                config.hidden_dim, 
                config.output_size,
                config.n_layers, 
                config.bidirectional)
    net.load_state_dict(torch.load(config.save_path))
    net.cuda()
    result_comments=pretreatment(test_comment_list)   #预处理去掉标点符号
    #转换为字id
    tokenizer = BertTokenizer.from_pretrained(config.bert_path)
    result_comments_id = tokenizer(result_comments,
                                    padding=True,
                                    truncation=True,
                                    max_length=120,
                                    return_tensors='pt')
    tokenizer_id = result_comments_id['input_ids']
    # print(tokenizer_id.shape)
    inputs = tokenizer_id
    batch_size = inputs.size(0)
    # batch_size = 32
    # initialize hidden state
    h = net.init_hidden(batch_size)

    if(USE_CUDA):
        inputs = inputs.cuda()

    net.eval()
    with torch.no_grad():
        # get the output from the model
        output= net(inputs, h)
        output=torch.nn.Softmax(dim=1)(output)
        pred=torch.max(output, 1)[1]
        # printing output value, before rounding
        print('预测概率为: {:.6f}'.format(torch.max(output, 1)[0].item()))
        if(pred.item()==1):
            print("预测结果为:正向")
        else:
            print("预测结果为:负向")



if __name__ == '__main__':
    model_config = ModelConfig()
    data=pd.read_csv('dianping.csv',encoding='utf-8')
    result_comments = pretreatment(list(data['comment'].values))
    tokenizer = BertTokenizer.from_pretrained(model_config.bert_path)

    result_comments_id = tokenizer(result_comments,
                                    padding=True,
                                    truncation=True,
                                    max_length=200,
                                    return_tensors='pt')
    X = result_comments_id['input_ids']
    y = torch.from_numpy(data['sentiment'].values).float()

    X_train,X_test, y_train, y_test = train_test_split( X,
                                                        y,
                                                        test_size=0.3,
                                                        shuffle=True,
                                                        stratify=y,
                                                        random_state=0)
    X_valid,X_test,y_valid,y_test = train_test_split(X_test,
                                                     y_test,
                                                     test_size=0.5,
                                                     shuffle=True,
                                                     stratify=y_test,
                                                     random_state=0)
    train_data = TensorDataset(X_train, y_train)
    valid_data = TensorDataset(X_valid, y_valid)
    test_data = TensorDataset(X_test,y_test)
    train_loader = DataLoader(train_data,
                                shuffle=True,
                                batch_size=model_config.batch_size,
                                drop_last=True)
    valid_loader = DataLoader(valid_data,
                                shuffle=True,
                                batch_size=model_config.batch_size,
                                drop_last=True)
    test_loader = DataLoader(test_data, 
                                shuffle=True, 
                                batch_size=model_config.batch_size,
                                drop_last=True)
    if(USE_CUDA):
        print('Run on GPU.')
    else:
        print('No GPU available, run on CPU.')
    # train_model(model_config, train_loader)
    test_model(model_config, test_loader)
    test_comments =  ['这个菜真不错']
    predict(test_comments, model_config)
    

