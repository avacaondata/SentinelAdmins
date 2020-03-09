library(ggplot2)
library(data.table)
install.packages('plotly')
library(plotly)

df = read.csv('Modelar_UH2020.txt', sep='|')


p = ggplot(df, aes(x=X, y=Y, colour=CONTRUCTIONYEAR)) +
  geom_point(size=0.1) +
  scale_colour_continuous(type='viridis')

ggplotly(p)