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
listed_data = listed_data.str.replace(' ו',' ')
listed_data = listed_data.str.replace('ת.','ת')
listed_data = listed_data.str.replace('Hebrew','עברית')
listed_data = listed_data.str.replace('hebrew','עברית')
listed_data = listed_data.str.replace('English','אנגלית')
listed_data = listed_data.str.replace('english','אנגלית')
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
    languages_count['Hebrew'][ind] = ('עברית' in set_answer) or ('עיברית' in set_answer)
    languages_count['English'][ind] = 'אנגלית' in set_answer
    languages_count['Spanish'][ind] = 'ספרדית' in set_answer
    languages_count['Russian'][ind] = 'רוסית' in set_answer
    languages_count['French'][ind] = 'צרפתית' in set_answer
    languages_count['Arabic'][ind] = 'ערבית' in set_answer
    languages_count['Italian'][ind] = 'איטלקית' in set_answer
    languages_count['German'][ind] = 'גרמנית' in set_answer
    languages_count['Portuguese'][ind] = ('פורטוגזית' in set_answer) or ('פורטגזית' in set_answer) or ('פורטוגסית' in set_answer)
    languages_count['Romanian'][ind] = 'רומנית' in set_answer
    languages_count['Ukrainian'][ind] = 'אוקראינית' in set_answer
    languages_count['Yiddish'][ind] = ('יידיש' in set_answer) or ('אידיש' in set_answer)
    languages_count['Polish'][ind] = 'פולנית' in set_answer
    languages_count['Chinese'][ind] = ('סינית' in set_answer) or ('מנדרינית' in set_answer) or ('קנטונזית' in set_answer)    
    languages_count['Greece'][ind] = 'יוונית' in set_answer
    languages_count['Parsian'][ind] = 'פרסית' in set_answer
    languages_count['Hungarian'][ind] = 'הונגרית' in set_answer
    languages_count['Catalan'][ind] = ('קטלאנית' in set_answer) or ('קטלונית' in set_answer)
    languages_count['Japanese'][ind] = 'יפנית' in set_answer
    languages_count['Swedish'][ind] = 'שוודית' in set_answer
    languages_count['Slovak'][ind] = 'סלובקית' in set_answer
    languages_count['Dutch'][ind] = 'הולנדית' in set_answer
    languages_count['Czech'][ind] = ('צ?כית' in set_answer) or ('צכית' in set_answer)
    languages_count['Latin'][ind] = 'לטינית' in set_answer
    languages_count['Georgian'][ind] = ('גאורגית' in set_answer) or ('גיאורגית' in set_answer)
    languages_count['Bulgarian'][ind] = 'בולגרית' in set_answer
    languages_count['Croatian'][ind] = 'קרואטית' in set_answer
    languages_count['Amharic'][ind] = 'אמהרית' in set_answer
    languages_count['Moldovan'][ind] = 'מולדובית' in set_answer
    
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
    if word.find('ית')==len(word)-2 & word.find('ית')>0:
        all_languages.append(word)