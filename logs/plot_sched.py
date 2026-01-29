f0 = 120
b0 = 246
f2 = 120
b2 = 244
up0 = 40

f1 = 120
b1 = 245
f3 = 183
b3 = 425
up1 = 39

start = 0
f00s = start
f00e = f00s + f0

f01s = f00e
f01e = f01s + f0

f10s = f00e
f10e = f10s + f1

f11s = max(f10e, f01e)
f11e = f11s + f1

f20s = max(f01e, f10e)
f20e = f20s + f2

f21s = max(f11e, f20e)
f21e = f21s + f2

f30s = max(f20e, f11e)
f30e = f30s + f3

b30s = f30e
b30e = b30s + b3

f02s = f21e
f02e = f02s + f0

f03s = f02e
f03e = f03s + f0

f31s = max(b30e, f21e)
f31e = f31s + f3

b31s = f31e
b31e = b31s + b3

b20s = max(f03e, b30e)
b20e = b20s + b2

f12s = max(b31e, f02e)
f12e = f12s + f1

b10s = max(f12e, b20e)
b10e = b10s + b1

b21s = max(b20e, b31e)
b21e = b21s + b2

f22s = max(b21e, f12e)
f22e = f22s + f2

f13s = max(b10e, f03e)
f13e = f13s + f1

b00s = max(f22e, b10e)
b00e = b00s + b0

b11s = max(f13e, b21e)
b11e = b11s + b1

f23s = max(b00e, f13e)
f23e = f23s + f2

f32s = max(b11e, f22e)
f32e = f32s + f3

b32s = f32e
b32e = b32s + b3

b01s = max(f23e, b11e)
b01e = b01s + b0

f33s = max(b32e, f23e)
f33e = f33s + f3

b33s = f33e
b33e = b33s + b3

b22s = max(b01e, b32e)
b22e = b22s + b2

b12s = max(b33e, b22e)
b12e = b12s + b1

b23s = max(b22e, b33e)
b23e = b23s + b2

b02s = max(b23e, b12e)
b02e = b02s + b0

b13s = max(b12e, b23e)
b13e = b13s + b1

b03s = max(b02e, b13e)
b03e = b03s + b0

u0s = b03e
u0e = u0s + up0

u1s = b13e
u1e = u1s + up1

end = max(u0e, u1e)
print(end)
