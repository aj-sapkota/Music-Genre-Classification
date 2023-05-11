from .model_redefine import makeTheModel
import torch


def predictingOutput(mfccCoeff):
    mappings = ['disco', 'metal', 'reggae', 'blues', 'rock',
                'classical', 'jazz', 'hiphop', 'country', 'pop']
    temp = mfccCoeff.shape[0]
    print(temp)
    print(mfccCoeff.shape)

    dataT = torch.tensor(mfccCoeff).float()
    dataT /= torch.max(dataT)

    mfccCoeff = dataT.view([temp, 1, 216, 13]).float()

    # predicting part
    # load model

    path = 'musicgenre/model.pth'
    device = torch.device('cpu')

    model = makeTheModel()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    # print('Model state dict')
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    with torch.no_grad():
        genrePredictionTemp = model(mfccCoeff)

    tempAverage = torch.sum(genrePredictionTemp, dim=0)
    tempAverage /= temp
    tempAverage = tempAverage.view([1, 10]).float()

    # print(tempAverage)

    predicted = torch.argmax(genrePredictionTemp, axis=1)
    #print(predicted)
    predicted = torch.argmax(tempAverage, axis=1)
    # print(predicted)
    # print(mappings)
    genreLabelling = mappings[predicted]
    # print(genreLabelling)
    return(genreLabelling,tempAverage)
    
