# Function to calculate first WAP
def calc_wap1(df):
    wap = (df['bp1'] * df['av1'] + df['ap1'] * df['bv1']) / (df['bv1'] + df['av1'])
    return wap

# Function to calculate second WAP
def calc_wap2(df):
    wap = (df['bp2'] * df['av2'] + df['ap2'] * df['bv2']) / (df['bv2'] + df['av2'])
    return wap

# Function to aggregate 1st and 2nd WAP
def calc_wap12(df):
    var1 = df['bp1'] * df['av1'] + df['ap1'] * df['bv1']
    var2 = df['bp2'] * df['av2'] + df['ap2'] * df['bv2']
    den = df['bv1'] + df['av1'] + df['bv2'] + df['av2']
    return (var1+var2) / den

def calc_wap3(df):
    wap = (df['bp1'] * df['bv1'] + df['ap1'] * df['av1']) / (df['bv1'] + df['av1'])
    return wap

def calc_wap34(df):
    var1 = df['bp1'] * df['bv1'] + df['ap1'] * df['av1']
    var2 = df['bp2'] * df['bv2'] + df['ap2'] * df['av2']
    den = df['bv1'] + df['av1'] + df['bv2'] + df['av2']
    return (var1+var2) / den

def calc_swap1(df):
    return calc_wap1(df) - calc_wap3(df)

def calc_swap12(df):
    return calc_wap12(df) - calc_wap34(df)

def calc_tswap1(df):
    return calc_swap1(df).diff().fillna(0)

def calc_tswap12(df):
    return calc_swap12(df).diff().fillna(0)