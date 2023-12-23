#### Analyze Languages Statistics ####

# modouls and read data from excel file
import pandas as pd
import numpy as np
data = pd.read_excel('BB_answers.xlsx')
data = data.fillna('empty')

# replace and fix data
listed_data = data['L2'].str.replace(',',' ')
listed_data = listed_data.str.replace('/',' ')
listed_data = listed_data.str.replace('-',' ')
listed_data = listed_data.str.replace('  ',' ')
listed_data = listed_data.str.replace(' �',' ')
listed_data = listed_data.str.replace('�.','�')
listed_data = listed_data.str.replace('Hebrew','�����')
listed_data = listed_data.str.replace('hebrew','�����')
listed_data = listed_data.str.replace('English','������')
listed_data = listed_data.str.replace('english','������')
listed_data = listed_data.str.split(' ')

# create a set of languages for each subject
set_data = {}
for answer,ind in zip(listed_data, range(len(listed_data))):
    if type(answer) == float:
        answer = [str(answer)]
    answer.append(data['L1'][ind])
    set_data[ind] = set(answer)

# count the number of subjects that answered in each language
languages_count = pd.DataFrame(columns=['Hebrew','English','Spanish','Russian','French','Arabic','Italian','German','Portuguese','Romanian','Ukrainian','Yiddish','Polish','Chinese','Greece', 'Parsian','Hungarian','Catalan','Japanese','Slovak','Swedish','Dutch','Czech','Latin','Georgian','Bulgarian','Croatian','Amharic','Moldovan'],index=range(len(set_data)))
for set_answer,ind in zip(set_data.values(),range(len(set_data))):
    languages_count['Hebrew'][ind] = ('�����' in set_answer) or ('������' in set_answer)
    languages_count['English'][ind] = '������' in set_answer
    languages_count['Spanish'][ind] = '������' in set_answer
    languages_count['Russian'][ind] = '�����' in set_answer
    languages_count['French'][ind] = '������' in set_answer
    languages_count['Arabic'][ind] = '�����' in set_answer
    languages_count['Italian'][ind] = '�������' in set_answer
    languages_count['German'][ind] = '������' in set_answer
    languages_count['Portuguese'][ind] = ('���������' in set_answer) or ('��������' in set_answer) or ('���������' in set_answer)
    languages_count['Romanian'][ind] = '������' in set_answer
    languages_count['Ukrainian'][ind] = '���������' in set_answer
    languages_count['Yiddish'][ind] = ('�����' in set_answer) or ('�����' in set_answer)
    languages_count['Polish'][ind] = '������' in set_answer
    languages_count['Chinese'][ind] = ('�����' in set_answer) or ('��������' in set_answer) or ('��������' in set_answer)    
    languages_count['Greece'][ind] = '������' in set_answer
    languages_count['Parsian'][ind] = '�����' in set_answer
    languages_count['Hungarian'][ind] = '�������' in set_answer
    languages_count['Catalan'][ind] = ('�������' in set_answer) or ('�������' in set_answer)
    languages_count['Japanese'][ind] = '�����' in set_answer
    languages_count['Swedish'][ind] = '������' in set_answer
    languages_count['Slovak'][ind] = '�������' in set_answer
    languages_count['Dutch'][ind] = '�������' in set_answer
    languages_count['Czech'][ind] = ('�?���' in set_answer) or ('����' in set_answer)
    languages_count['Latin'][ind] = '������' in set_answer
    languages_count['Georgian'][ind] = ('�������' in set_answer) or ('��������' in set_answer)
    languages_count['Bulgarian'][ind] = '�������' in set_answer
    languages_count['Croatian'][ind] = '�������' in set_answer
    languages_count['Amharic'][ind] = '������' in set_answer
    languages_count['Moldovan'][ind] = '��������' in set_answer
    
# sum the number of languages each subject answered in
languages_count['Total'] = languages_count.sum(axis=1)
five_and_above = np.where(languages_count['Total'] >= 5)
four = np.where(languages_count['Total'] == 4)

# count the number of subjects that answered in each language
summary = languages_count.sum(axis=0)

### check for more languages
all_words = []
all_languages = []
for answer in set_data.values():
    all_words.extend(list(answer))
all_words = set(all_words)
for word in all_words:
    if word.find('��')==len(word)-2 & word.find('��')>0:
        all_languages.append(word)