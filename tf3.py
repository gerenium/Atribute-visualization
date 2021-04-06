import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from matplotlib import cm
from colorspacious import cspace_converter
from collections import OrderedDict

cmap = OrderedDict()                  #  получение цветовой карты

large = 20; med = 14; small = 10      # размеры для стандартизации окна
size_title =26; size_label = 16; width_line = 2

params = {'axes.titlesize': large,      # параметры для отрисовки полотна
          'legend.fontsize': med,
          'figure.figsize': (med, 8),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')

# Import Dataset
df1 = pd.read_csv("H:\\Downloads\\117fa20b-d087-487c-b19f-464b27a79ed5 (1).csv")
print(df1)
df1 = df1[df1["Year_of_Release"] < 2017] # отсечение того единственного выброса

columns = df1.columns.values
#['Name' 'Platform' 'Year_of_Release' 'Genre' 'Publisher' 'NA_Sales'
# 'EU_Sales' 'JP_Sales' 'Other_Sales' 'Global_Sales' 'Critic_Score'
# 'Critic_Count' 'User_Score' 'User_Count' 'Developer' 'Rating']

df1 = df1.dropna(subset = ['Name']).reset_index(drop = True)                                # удаляем строки с пустыми значениями, так как их не много, а с данными работать удобнее
df1 = df1.dropna(subset = ['Year_of_Release']).reset_index(drop = True)
df1 = df1.dropna(subset = ['Publisher']).reset_index(drop = True)
df1.info()

group_data_1 = df1.groupby('Year_of_Release')['Name'].count()           # получения и сортировка уникальных значений из разных столбцов для понимания с чем предстоит работать
years = df1['Year_of_Release'].unique()
years.sort()
genres = df1['Genre'].unique()
genres.sort()
plats = df1['Platform'].unique()
plats.sort()
# график количества игр по годам
frame2 = pd.DataFrame(group_data_1, columns = years)                                                              # построения 1го графика
plt.plot(years, group_data_1, color="g", linewidth= width_line + 1)
plt.xlabel(u'Года', fontsize= size_label)
plt.ylabel(u'Количество вышедших игр', fontsize= size_label)
plt.title('Изменение количества вышедших игр', fontsize= size_title, fontweight = 'heavy' )
plt.show()

#=======================================================================================================================

#=======================================================================================================================
group_data_2 = df1.groupby('Platform')['Name'].count()          # сбор и обработка данных
frame2 = pd.DataFrame(zip(group_data_2, plats))
frame2.columns=['0','1']
frame2 = frame2.sort_values(by = '0')

# линейчатая диаграма популярности платформ
sns.set_style("white")
plt.title('Популярность платформ, на которых выходили игры', fontsize= size_title, fontweight = 'heavy' )        # построение 2го графика
plt.barh(frame2['1'], frame2['0'], height=0.7, color="#ffa500")
plt.xlabel(u'Количество вышедших игр на платформе', fontsize= size_label)
plt.ylabel(u'Название платформы', fontsize= size_label)
plt.show()

#=======================================================================================================================

#=======================================================================================================================
NA_sum = df1['NA_Sales'].sum()                          # сбор и обработка данных
EU_sum = df1['EU_Sales'].sum()
JP_sum = df1['JP_Sales'].sum()
Ot_sum = round(df1['Other_Sales'].sum(), 2)

valvs = [NA_sum, EU_sum, JP_sum, Ot_sum]
labels = ['Северная Америка', 'Европа', 'Япония', 'Остальные страны']

# круговая диаграмма о суммарных продажах 
cmap = plt.get_cmap('nipy_spectral',10)                                                                          # построение 3го графика
b_colors = cmap(np.array([3, 4, 7, 22]))
fig, ax = plt.subplots()
ax.pie(valvs, labels=labels,autopct='%1.1f%%',  textprops={'fontsize': 18}, startangle = 90, colors = b_colors)
plt.title('Отношение суммарных продаж в разных регионах', fontsize= size_title, fontweight = 'heavy' )
plt.show()

#=======================================================================================================================

#=======================================================================================================================
df2 = df1.copy()
df2['User_Score'] = df2['User_Score'].replace('tbd', 0)
df2['User_Score'] = df2['User_Score'].astype(float)
 
c_score_mean = round(df1.groupby('Genre')['Critic_Score'].mean()/10, 2)
u_score_mean = df2.groupby('Genre')['User_Score'].mean()

frame3 = pd.DataFrame(zip(genres , c_score_mean, u_score_mean))
frame3.columns=[0, 1, 2]

ax = plt.axes()
ax.yaxis.grid(True, zorder = 1)

# линейчата диаграмма оценки жанров
a_colors = cmap(np.array([3]))                                                                                     # построение 4го графика
xs = range(len(genres))
plt.bar(frame3[0], frame3[1],
        width = 0.3, color = 'orange', alpha = 0.8, label = 'Оценка критиками',
        zorder = 2)
plt.bar([x + 0.3 for x in xs], frame3[2],
        width = 0.3, color = a_colors, alpha = 0.9, label = 'Оценка пользователями',
        zorder = 2)
#plt.legend(loc=1, mode='expand', numpoints=1, ncol=4, fancybox = True,
#           fontsize='small', labels=['d1', 'd2'])
#plt.legend(bbox_to_anchor=(1, 0.1))
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=3)
#plt.axis('off') 
#plt.xticks([])
plt.title('Средняя оценка жанров', fontsize= size_title, fontweight = 'heavy' )
plt.show()

#=======================================================================================================================

#=======================================================================================================================
rat = df2.groupby('Rating')['Name'].count()
df2['Rating'] = df2['Rating'].replace('K-A', 'E')
df2['Rating'] = df2['Rating'].replace('AO', 'M')           # удаляем слишком редкие категории, причисляя их к ближайшей крупной/ актуальной                         
df2['Rating'] = df2['Rating'].replace('EC', 'E')
df2 = df2[~df2['Rating'].isin(['RP'])]
rating = df2.groupby('Rating')['Name'].count()

valvs = rating
labels = ['от 6 лет', 'от 10 лет', 'от 17 лет', 'от 13 лет']

# круговая диаграмма возрастных категорий
cmap = plt.get_cmap('nipy_spectral',10)                                                                             # построение 5го графика
b_colors = cmap(np.array([3, 4, 7, 10]))
fig, ax = plt.subplots()
ax.pie(valvs, labels=labels,autopct='%1.1f%%',  textprops={'fontsize': 18}, startangle = 90, colors = b_colors)
plt.title('Соотношение возрастных категорий, в которых выходили игры', fontsize= size_title, fontweight = 'heavy' )
plt.show()

#=======================================================================================================================

#=======================================================================================================================
NA_sales = df1.groupby('Year_of_Release')['NA_Sales'].sum()
EU_sales = df1.groupby('Year_of_Release')['EU_Sales'].sum()
JP_sales = df1.groupby('Year_of_Release')['JP_Sales'].sum()
O_sales = df1.groupby('Year_of_Release')['Other_Sales'].sum()
G_sales = df1.groupby('Year_of_Release')['Global_Sales'].sum()

frame4 = pd.DataFrame(zip(years, NA_sales, EU_sales, JP_sales, O_sales, G_sales))
frame4.columns=[0, 1, 2, 3, 4, 5]

# график изменения продаж по регионам
plt.plot(frame4[0], frame4[1], label = 'Продажи в Северной Америке', linewidth= width_line)                       # построение 6го графика
plt.plot(frame4[0], frame4[2], label = 'Продажи в Европе', linewidth= width_line)
plt.plot(frame4[0], frame4[3], label = 'Продажи в Японии', linewidth= width_line)
plt.plot(frame4[0], frame4[4], label = 'Продажи в остальных странах',linewidth= width_line)
plt.plot(frame4[0], frame4[5], label = 'Продаж всего',linewidth= width_line)
plt.title('Изменение объема продаж в разных странах', fontsize=size_title, fontweight = 'heavy' )
plt.legend(loc = 'upper left')
plt.show()

#=======================================================================================================================

#=======================================================================================================================
plt.title('Зависимость оценки и продаж', fontsize= size_title, fontweight = 'heavy' )
plt.scatter(df2['Global_Sales'], df2['User_Score'])
plt.show()

#=======================================================================================================================

#=======================================================================================================================
df2 = df2[df2['User_Score'] > 0] 
plt.title('Корреляция оценок критиков и оценок пользователей', fontsize= size_title, fontweight = 'heavy' )
plt.scatter(df2['Critic_Score']/10, df2['User_Score'], s = 4)
plt.xlabel(u'Оценка критиков', fontsize= size_label)
plt.ylabel(u'Оценка пользователей', fontsize= size_label)
plt.show()