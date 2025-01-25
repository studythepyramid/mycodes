
from decimal import *
getcontext().prec = 100  # Change the precision
d = Decimal(2).sqrt()
print (d)

#Decimal('1.414213562373095048801688724209698078569671875376948073176679737990732478462107038850387534327641573')


## another lib
from bigfloat import *
sqrt(2, precision(100))  # compute sqrt(2) with 100 bits of precision

