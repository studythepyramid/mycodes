import numpy as np
import pandas as pd
import math
import plotly.express as px


#df = px.data.gapminder().query("country=='Canada'")

x = np.linspace(start= -2.0, stop=2.0, num=100)

col = 1.0 / (math.sqrt(2.0 * math.pi))

y = [ col * math.exp(-0.5 * d * d) for d in x]

df = pd.DataFrame(data={"xData": x, "yValue": y})

fig = px.line(df, x="xData", y="yValue", title='Guassian Distribution')
fig.show()
