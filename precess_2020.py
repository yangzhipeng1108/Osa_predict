import  pandas as pd
import numpy  as np

df = pd.read_csv('shhs_inter.csv')

def product_BMI(x):

    return  x['weight'] / ((x['height'] / 100) * (x['height'] / 100))

df['BMI'] = df.apply(lambda x: product_BMI(x),axis = 1)

def product_LAP(x):
    if x['gender'] == 1:
        return (x['wc'] - 65) * x['TG'] if x['wc'] > 65 else x['TG']
    elif x['gender'] == 2:
        return (x['wc'] - 58) * x['TG'] if x['wc'] > 58 else x['TG']
df['LAP'] = df.apply(lambda x: product_LAP(x),axis = 1)

def product_VAI(x):
    if x['gender'] == 1:
        return (x['wc'] / (39.68 + (1.88 * x['BMI']))) * (x['TG'] / 1.03) * (1.31 / x['HDL'])
    elif x['gender'] == 2:
        return (x['wc'] / (36.58 + (1.89 * x['BMI']))) * (x['TG'] / 0.81) * (1.52 / x['HDL'])

df['VAI'] = df.apply(lambda x: product_VAI(x),axis = 1)

def product_CVAI(x):
    if x['gender'] == 1:
        return -267.93 + 0.68*x['age_diagnose'] + 0.03 * x['BMI'] + 4.00 * x['wc'] + 22.00 * np.log10(x['TG']) - 16.32 * x['HDL']
    elif x['gender'] == 2:
        return -187.32 + 1.71*x['age_diagnose'] + 4.23 * x['BMI'] + 1.12 * x['wc'] + 39.76 * np.log10(x['TG']) - 11.66 * x['HDL']

df['CVAI'] = df.apply(lambda x: product_CVAI(x),axis = 1)

def product_ABSI(x):

    return (x['wc'] / 100) / ((x['BMI'] ** (2/3)) * ((x['height'] / 100)** (1/2)))

df['ABSI'] = df.apply(lambda x: product_ABSI(x),axis = 1)

def product_BRI(x):

    return 364.2 - 365.5 * np.sqrt(1-( (((x['wc'] / 100) / (2 * np.pi)) ** 2) / ((0.5 * (x['height'] / 100)) **2)))

df['BRI'] = df.apply(lambda x: product_BRI(x),axis = 1)

def product_BAE(x):

    return -44.988 + (0.503 * x['age_diagnose']) + (10.689 * (x['gender'] - 1)) + (3.172 * x['BMI']) - (0.026 * (x['BMI'] ** 2)) \
             + (0.181 * x['BMI'] * (x['gender'] - 1)) - (0.02 * x['BMI'] * x['age_diagnose']) - (0.005 * (x['BMI'] **2) * (x['gender'] - 1)) \
          + (0.00021 * x['BMI'] * x['BMI'] * x['age_diagnose'])

df['BAE'] = df.apply(lambda x: product_BAE(x),axis = 1)

def product_PI(x):

    return  x['weight'] / ((x['height'] /100)** 3)

df['PI'] = df.apply(lambda x: product_PI(x),axis = 1)

def product_RFM(x):

    return  64 - (20 * (x['height'] / x['wc'])) + (12 *(x['gender']- 1 ))

df['RFM'] = df.apply(lambda x: product_RFM(x),axis = 1)

def product_CI(x):

    return  (x['wc'] /100) / 0.109 * np.sqrt(x['weight'] / (x['height'] / 100))

df['CI'] = df.apply(lambda x: product_CI(x),axis = 1)

def product_AVI(x):

    return  (2 * (x['wc'] ** 2) + 0.7 * (( x['wc'] - x['tunwei']) ** 2) )/1000

df['AVI'] = df.apply(lambda x: product_AVI(x),axis = 1)



def product_WHR(x):

    return  x['wc'] / (x['tunwei'])

df['WHR'] = df.apply(lambda x: product_WHR(x),axis = 1)

def product_WHtR(x):

    return  x['wc'] / (x['height'])

df['WHtR'] = df.apply(lambda x: product_WHtR(x),axis = 1)

def product_Lean_body_mass(x):
    if x['gender'] == 1:
        if x['race'] == 2 :
            return 19.363 + 0.001 * x['age_diagnose'] + 0.064* x['height']  + 0.756 * x['weight'] -0.366 * x['wc']   + 0.432 * 1

        elif x['ethnicity'] == 1 :
            return 19.363 + 0.001 * x['age_diagnose'] + 0.064 * x['height'] + 0.756 * x['weight'] - 0.366 * x[
                'wc'] + 0.231 * 1
        else:
            return 19.363 + 0.001 * x['age_diagnose'] + 0.064* x['height']  + 0.756 * x['weight'] -0.366 * x['wc']  - 1.007* 1
    elif x['gender'] == 2:
        if x['race'] == 2 :
            return -10.683 - 0.039 * x['age_diagnose'] + 0.186 * x['height'] + 0.383 * x['weight'] - 0.043 * x[
                'wc'] + 1.085 * 1
        elif x['ethnicity'] == 1 :
            return -10.683 - 0.039 * x['age_diagnose'] + 0.186 * x['height'] + 0.383 * x['weight'] - 0.043 * x[
                'wc'] - 0.059 * 1
        else:
            return -10.683 - 0.039 * x['age_diagnose'] + 0.186 * x['height'] + 0.383 * x['weight'] - 0.043 * x[
                'wc'] - 0.34 * 1


df['Lean_body_mass'] = df.apply(lambda x: product_Lean_body_mass(x),axis = 1)

def product_Fat_mass(x):
    if x['gender'] == 1:
        if x['race'] == 2 :
            return -18.592 - 0.009 * x['age_diagnose'] - 0.080* x['height']  + 0.226 * x['weight'] +0.387 * x['wc']  - 0.483 *1

        elif x['ethnicity'] == 1 :
            return -18.592 - 0.009 * x['age_diagnose'] - 0.080* x['height']  + 0.226 * x['weight'] +0.387 * x['wc']   - 0.188 * 1
        else:
            return -18.592 - 0.009 * x['age_diagnose'] - 0.080* x['height']  + 0.226 * x['weight'] +0.387 * x['wc']  + 1.050*1
    elif x['gender'] == 2:
        if x['race'] == 2 :
            return 11.817 + 0.041 * x['age_diagnose'] - 0.199* x['height']  + 0.610 * x['weight'] +0.044 * x['wc']  -1.187 *1
        elif x['ethnicity'] == 1 :
            return 11.817 + 0.041 * x['age_diagnose'] - 0.199* x['height']  + 0.610 * x['weight'] +0.044 * x['wc'] + 0.073*1
        else:
            return 11.817 + 0.041 * x['age_diagnose'] - 0.199* x['height']  + 0.610 * x['weight'] +0.044 * x['wc'] + 0.325*1

df['Fat_mass'] = df.apply(lambda x: product_Fat_mass(x),axis = 1)

def product_Percent_fat(x):
    if x['gender'] == 1:
        if x['race'] == 2 :
            return 0.02 + 0.00 * x['age_diagnose'] - 0.07* x['height'] - 0.08 * x['weight'] +0.48 * x['wc'] - 0.65 *1

        elif x['ethnicity'] == 1 :
            return 0.02 + 0.00 * x['age_diagnose'] - 0.07* x['height'] - 0.08 * x['weight'] +0.48 * x['wc'] +  0.02 *1
        else:
            return 0.02 + 0.00 * x['age_diagnose'] - 0.07* x['height'] - 0.08 * x['weight'] +0.48 * x['wc'] +  1.12*1
    elif x['gender'] == 2:
        if x['race'] == 2 :
            return 50.46 +  0.07 * x['age_diagnose'] - 0.26* x['height']  + 0.27 * x['weight'] +0.10 * x['wc'] - 1.57 *1
        elif x['ethnicity'] == 1 :
            return 50.46 +  0.07 * x['age_diagnose'] - 0.26* x['height']  + 0.27 * x['weight'] +0.10 * x['wc'] + 0.49*1
        else:
            return 50.46 +  0.07 * x['age_diagnose'] - 0.26* x['height']  + 0.27 * x['weight'] +0.10 * x['wc'] + 0.43*1

df['Percent_fat'] = df.apply(lambda x: product_Percent_fat(x),axis = 1)

print(df)

df.to_csv('shhs_data.csv',index = False)