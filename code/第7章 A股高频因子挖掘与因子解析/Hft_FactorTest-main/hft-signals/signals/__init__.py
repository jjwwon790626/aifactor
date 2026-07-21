def calc_dispersion(df):
    bspread = df['bp1'] - df['bp2']
    aspread = df['ap2'] - df['ap1']
    bmid = (df['bp1'] + df['ap1'])/2  - df['bp1']
    bmid2 = (df['bp1'] + df['ap1'])/2  - df['bp2']
    amid = df['ap1'] - (df['bp1'] + df['ap1'])/2
    amid2 = df['ap2'] - (df['bp1'] + df['ap1'])/2
    bdisp = (df['bv1']*bmid + df['bv2']*bspread)/(df['bv1']+df['bv2'])
    bdisp2 = (df['bv1']*bmid + df['bv2']*bmid2)/(df['bv1']+df['bv2'])
    adisp = (df['av1']*amid + df['av2']*aspread)/(df['av1']+df['av2'])      
    adisp2 = (df['av1']*amid + df['av2']*amid2)/(df['av1']+df['av2'])
    return bspread, aspread, bmid, amid, bdisp, adisp, (bdisp + adisp)/2, (bdisp2 + adisp2)/2