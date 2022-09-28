from CA import OneDimCA
import numpy as np

ca = OneDimCA(10, 28)

print(ca.state)
print(ca.state.increment_ca(28))