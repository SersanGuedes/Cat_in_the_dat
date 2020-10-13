def FNC_tratandoFeatures( df_titanic_2, aux_name, aux_cabin, aux_ticket ):
    if aux_name == 1:
        L_name = [("Mr." in x) or 
            ("Mrs." in x) or
            ("Miss." in x) or
            ("Ms." in x) or
            ("Sir." in x) or
            ("Madam." in x) or
            ("Lady." in x) or 
            ("Mlle." in x) for x in df_titanic_2["Name"]]
        
        df_titanic_2["Name"] = L_name

    elif aux_name == 2:
        L_name = [] #(!)Fazendo essa especificação maior, piorou acurácia.
        for x in df_titanic_2.Name:
            if ("Mr." in x):
                L_name.append("Mr.")
            elif ("Mrs." in x):
                L_name.append("Mrs.")
            elif ("Mrs." in x):
                L_name.append("Miss.")
            elif ("Mrs." in x):
                L_name.append("Ms.")
            elif ("Mrs." in x):
                L_name.append("Sir.")
            elif ("Mrs." in x):
                L_name.append("Madam.")
            elif ("Mrs." in x):
                L_name.append("Lady.")
            elif ("Mrs." in x):
                L_name.append("Mlle.")
            elif ("Mrs." in x):
                L_name.append("Mrs.")
            else:
                L_name.append("999")

        df_titanic_2["Name"] = L_name


    ##--- Tratando feature 'Cabin'
    if aux_cabin == 1:
        L_cabin = []
        for x in df_titanic_2.Cabin:
            if type(x)==float:
                # L_cabin.append(x)
                L_cabin.append("999") #(!)Acurácia permaneceu a mesma.
            elif type(x)==str:
                L_cabin.append(x[0])

        df_titanic_2["Cabin"] = L_cabin

    if aux_cabin == 2:
        L_cabin = [ 0 for x in df_titanic_2.Cabin ]

        df_titanic_2["Cabin"] = L_cabin


    ##--- Tratando feature 'Ticket'
    if aux_ticket == 1:
        L_ticket = [x[0] for x in df_titanic_2.Ticket ]

        df_titanic_2["Ticket"] = L_ticket

    elif aux_ticket == 2:
        L_ticket = [ 0 for x in df_titanic_2.Ticket ]

        df_titanic_2["Ticket"] = L_ticket

    return df_titanic_2