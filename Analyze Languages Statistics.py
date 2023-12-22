#### Analyze Languages Statistics ####

# modouls and read data from excel file
import pandas as pd
import numpy as np
data = pd.read_excel('BB_answers.xlsx')
data = data.fillna('empty')

# replace and fix data
listed_data = data['L2'].str.replace(',',' ')
listed_data = listed_data.str.replace('  ',' ')
listed_data = listed_data.str.replace(' ו',' ')
listed_data = listed_data.str.replace('Hebrew','עברית')
listed_data = listed_data.str.replace('hebrew','עברית')
listed_data = listed_data.str.replace('English','אנגלית')
listed_data = listed_data.str.replace('english','אנגלית')
listed_data = listed_data.str.split(' ')

# create a set of languages for each subject
set_data = {}
for answer,ind in zip(listed_data[0:1000],range(1000)):
    answer.append(data['L1'][ind])
    set_data[ind] = set(answer)

# count the number of subjects that answered in each language
languages_count = pd.DataFrame(columns=['Hebrew','English','Arabic','Spanish','Russian','French','Italian','German','Portuguese','Romanian','Ukrainian','Yiddish','Polish','Chinese','Greece'],index=range(1000))
for set_answer,ind in zip(set_data.values(),range(1000)):
    languages_count['Hebrew'][ind] = 'עברית' in set_answer
    languages_count['English'][ind] = 'אנגלית' in set_answer
    languages_count['Arabic'][ind] = 'ערבית' in set_answer
    languages_count['Spanish'][ind] = 'ספרדית' in set_answer
    languages_count['Russian'][ind] = 'רוסית' in set_answer
    languages_count['French'][ind] = 'צרפתית' in set_answer
    languages_count['Italian'][ind] = 'איטלקית' in set_answer
    languages_count['German'][ind] = 'גרמנית' in set_answer
    languages_count['Portuguese'][ind] = 'פורטוגזית' in set_answer
    languages_count['Romanian'][ind] = 'רומנית' in set_answer
    languages_count['Ukrainian'][ind] = 'אוקראינית' in set_answer
    languages_count['Yiddish'][ind] = ('יידיש' in set_answer) or ('אידיש' in set_answer)
    languages_count['Polish'][ind] = 'פולנית' in set_answer
    languages_count['Chinese'][ind] = ('סינית' in set_answer) or ('מנדרינית' in set_answer) or ('קנטונזית' in set_answer)    
    languages_count['Greece'][ind] = 'יוונית' in set_answer

# sum the number of languages each subject answered in
languages_count['Total'] = languages_count.sum(axis=1)
more_than_4 = np.where(languages_count['Total'] > 4)

# count the number of subjects that answered in each language
summary = languages_count.sum(axis=0)