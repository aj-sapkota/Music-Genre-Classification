# Online Python compiler (interpreter) to run Python online.
# Write Python 3 code in this online editor and run it.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

#from musicgenre.forGraph import forGraph


#mappings, Labellings = forGraph()


#labels = ['disco', 'metal', 'reggae', 'blues', 'rock', 'classical', 'jazz', 'hiphop', 'country', 'pop']

#abc = my_array.flatten()
#print(abc.shape)
#df = pd.DataFrame(abc, columns = ['Column_A'])

#print(df)
#print(type(df))

#create pandas DataFrame
def get_dataframe(tempAverage):
    my_array = tempAverage
    #my_array_max = my_array.max()
    normalized_value = preprocessing.normalize(my_array)
    print(normalized_value)
    df = pd.DataFrame({'Genres': ['disco', 'metal', 'reggae', 'blues', 'rock', 'classical', 'jazz', 'hiphop', 'country', 'pop'],
                    })

    #create NumPy array for 'blocks'
    #labels = ['disco', 'metal', 'reggae', 'blues', 'rock', 'classical', 'jazz', 'hiphop', 'country', 'pop']

    abc = normalized_value.flatten()
    #blocks = np.array([2, 3, 1, 0, 2, 7, 8, 2,1,1])

    #add 'blocks' array as new column in DataFrame
    list_abc = abc.tolist()
    df['Comparative_Value'] = list_abc
    

    #df.plot(kind='bar',x='Genres',y='tempAverage')
    #dfh = df.to_html()
    #print(dfh)
# the plot gets saved to 'output.png'
    #plt.savefig('./static/images/music.jpg')
    #df.plot(x='Genres',y='tempAverage')
#display the DataFrame
    return df



    