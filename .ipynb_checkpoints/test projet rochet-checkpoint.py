import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv("Downloads/emission-co2-perimetre-complet (1).csv", sep = ';')


df["Train"] = df['Train_Empreinte carbone (kgCO2e)'] / df["Distance entre les gares"]
df["Bus"] = df['Autocar longue distance - Empreinte carbone (kgCO2e)'] / df['Distance entre les gares']
df["Voiture thermique"] = df['Voiture thermique (2,2 pers.)_Empreinte carbone (kgCO2e)'] / df['Distance entre les gares']
df["Voiture electrique"] = df['Voiture électrique (2,2 pers.) - Empreinte carbone (kgCO2e)'] / df['Distance entre les gares']
df["Avion"] = df['Avion_Empreinte carbone (kgCO2e)'] / df["Distance entre les gares"]

df["Trajet"] = df["Origine"] + " - " + df["Destination"]

def nombre_occurence_pie():
    transporteur_mean = df['Transporteur'].value_counts()


    plt.figure(figsize=(8, 8))
    plt.pie(
        transporteur_mean.values,
        labels=transporteur_mean.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=sns.color_palette('viridis', len(transporteur_mean))
    )

    plt.title('Répartition des occurrences par transporteur', fontsize=14)

    plt.tight_layout()
    plt.show()

def trajets():

    fig = px.treemap(df, 
                  path=['Transporteur', 'Origine', 'Destination'], 
                  title='Émissions des trajets par transporteur et gares')

    fig.show()

def emission_moyenne():
    emissions_par_transporteur = df.groupby("Transporteur")["Train"].mean()
    emissions_par_transporteur_df = emissions_par_transporteur.reset_index()
    emissions_par_transporteur_df.columns = ["Transporteur", "Moyenne des émissions"]

    autocar = pd.DataFrame({
        "Transporteur": ["Autocar"],
        "Moyenne des émissions": [df["Bus"].mean()]
    })

    voitelec = pd.DataFrame({
        "Transporteur": ["Voiture electrique"],
        "Moyenne des émissions": [df["Voiture electrique"].mean()]
    })

    voither = pd.DataFrame({
        "Transporteur": ["Voiture thermique"],
        "Moyenne des émissions": [df["Voiture thermique"].mean()]
    })
        
    emissions_par_transporteur_df = pd.concat([emissions_par_transporteur_df, autocar], ignore_index=True)
    emissions_par_transporteur_df = pd.concat([emissions_par_transporteur_df, voitelec], ignore_index=True)
    emissions_par_transporteur_df = pd.concat([emissions_par_transporteur_df, voither], ignore_index=True)
    print(emissions_par_transporteur_df)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=emissions_par_transporteur_df["Transporteur"],
        y=emissions_par_transporteur_df["Moyenne des émissions"],
        palette="YlOrRd"
    )

    plt.title("Émissions calculées par transporteur", fontsize=16)
    plt.xlabel("Transporteur", fontsize=14)
    plt.ylabel("Émissions (kgCO2e / km)", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()

def emission_variance():
    emissions_par_transporteur = df.groupby("Transporteur")["Train"].var()

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=emissions_par_transporteur.index,
        y=emissions_par_transporteur.values,
        palette="Purples"
    )

    plt.title("Émissions calculées par transporteur", fontsize=16)
    plt.xlabel("Transporteur", fontsize=14)
    plt.ylabel("Émissions (kgCO2e / km)", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()

def avion_emission():

    vols_internationaux = ["Bruxelles Midi Brussel Zuid", "London St-Pancras", "Frankfurt Main HBF", "Milano P Garibaldi", "Amsterdam Centraal", "Barcelona Sants", "Luxembourg", "Torino Porta Susa", "Zuerich HB", "Muenchen HBF", "Geneve", "Schiphol Airport", "Stuttgart HBF", "Bruxelles N Aero", "Lausanne", "Basel SBB"]

    df["Vol International"] = df.apply(lambda x: "Oui" if x["Origine"] in vols_internationaux or x["Destination"] in vols_internationaux else "Non", axis=1)
    
    results = df.loc[df["Avion"].notnull(), ["Origine", "Destination", "Distance entre les gares", "Avion", "Vol International"]]

    print(results.sort_values(by="Avion", ascending=False))

    sns.scatterplot(
        data=results,
        x="Distance entre les gares",
        y="Avion",
        hue="Vol International",
        palette={"Oui": "red", "Non": "blue"},
        alpha=0.7
    )

    plt.title("Ratio d'émission avion en fonction de la distance (Vol International ou Non)")
    plt.xlabel("Distance entre les gares")
    plt.ylabel("Ratio d'émission avion")
    plt.legend(title="Vol International")
    plt.grid(True)
    plt.show()

def avion_train_voiture():

    df_long = df.loc[df["Avion"].notnull()]
    print(df_long[["Trajet", "Avion"]])


    df_long = df_long.melt(
        id_vars=["Trajet"],
        var_name="Mode de transport",
        value_name="Émissions (kgCO2e / km)"
    )

    df_long = df_long.loc[df_long["Mode de transport"].isin(["Train", "Bus", "Voiture electrique", "Voiture thermique", "Avion"])]
    ordre = ["Train", "Bus", "Voiture electrique", "Voiture thermique", "Avion"]

    sns.set(rc={'figure.autolayout': False})
    sns.set(style="whitegrid")

    g = sns.catplot(
        data=df_long,
        x="Mode de transport",
        y="Émissions (kgCO2e / km)",
        kind="violin",
        height=6,
        aspect=1.5,
        palette="viridis",
        inner=None,
        order=ordre
    )

    sns.swarmplot(data=df_long, x="Mode de transport", y="Émissions (kgCO2e / km)", size=3, color="white")

    g.set_xticklabels(rotation=0, fontsize=10)
    g.fig.suptitle("Émissions de gaz à effet de serre par trajet et mode de transport", fontsize=16)
    g.set_axis_labels("Trajet", "Émissions (kgCO2e / km)")

    plt.tight_layout()

    plt.show()

def train_voiture():

    df["Emission bus"] = df['Autocar longue distance - Empreinte carbone (kgCO2e)'] / df['Distance entre les gares']
    df["Voiture thermique"] = df['Voiture thermique (2,2 pers.)_Empreinte carbone (kgCO2e)'] / df['Distance entre les gares']
    df["Voiture electrique"] = df['Voiture électrique (2,2 pers.) - Empreinte carbone (kgCO2e)'] / df['Distance entre les gares']
    df["Trajet"] = df["Origine"] + " - " + df["Destination"]


    df_long = df.melt(
        id_vars=["Trajet"],
        var_name="Mode de transport",
        value_name="Émissions (kgCO2e / km)"
    )

    df_long = df_long.loc[df_long["Mode de transport"].isin(["Train", "Bus", "Voiture electrique", "Voiture thermique"])]
    ordre = ["Train", "Bus", "Voiture electrique", "Voiture thermique"]

    sns.set(rc={'figure.autolayout': False})
    sns.set(style="whitegrid")

    sns.stripplot(
        data=df_long,
        x="Mode de transport",
        y="Émissions (kgCO2e / km)",
        hue="Mode de transport",
        jitter=True,
        palette="viridis",
        order=ordre
    )
    plt.title("Émissions par Trajet et Mode de Transport", fontsize=16)
    plt.xlabel("Trajets", fontsize=12)
    plt.ylabel("Émissions (kgCO2e / km)", fontsize=12)
    plt.xticks(rotation=45)
    plt.show()

def ligne():
    df_long = df.melt(
        id_vars=["Distance entre les gares"],
        value_vars=["Train_Empreinte carbone (kgCO2e)", "Autocar longue distance - Empreinte carbone (kgCO2e)", "Avion_Empreinte carbone (kgCO2e)", "Voiture électrique (2,2 pers.) - Empreinte carbone (kgCO2e)", "Voiture thermique (2,2 pers.)_Empreinte carbone (kgCO2e)"],
        var_name="Mode de transport",
        value_name="Émissions (kgCO2e / km)"
    )

    plt.figure(figsize=(12, 8))
    sn = sns.lmplot(
        data=df_long, 
        x="Distance entre les gares", 
        y="Émissions (kgCO2e / km)", 
        hue="Mode de transport"
    )

    sn.legend.remove()
    plt.title("Émissions de CO2 par mode de transport en fonction de la distance", fontsize=16)
    plt.xlabel("Distance entre les gares (km)", fontsize=14)
    plt.ylabel("Émissions (kgCO2e / km)", fontsize=14)
    plt.legend(title="Mode de transport")
    plt.tight_layout()

    plt.show()
    

# nombre_occurence_pie()
# emission_variance()
# emission_moyenne()
# avion_emission()
trajets()
avion_train_voiture()
train_voiture()
ligne()
