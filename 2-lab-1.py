import numpy as np
import math
import random
import matplotlib.pyplot as plt

# Распределение энергии

tissue = [
  { 'ua': 32, 'us': 165, 'g': 0.72, 'd': 0.01 },
  { 'ua': 40, 'us': 246, 'g': 0.72, 'd': 0.02 },
  { 'ua': 23, 'us': 227, 'g': 0.72, 'd': 0.02 },
  { 'ua': 46, 'us': 253, 'g': 0.72, 'd': 0.06 },
  { 'ua': 23, 'us': 227, 'g': 0.72, 'd': 0.09 },
]

maxD = sum(map(lambda x: x['d'], tissue))

# Данные для удобного счета графика тепловой карты
normFactor = 1000;
xDelta = 0.05 * normFactor;
minX = int(-xDelta)
maxX = int(xDelta)
minZ = 0
maxZ = int(maxD * 1.1 * normFactor)
segmentDelta = int(0.001 * normFactor)

# Создаем словарь для хранения поглощенной энергии в каждом сегменте
segment_energy = {}

# Инициализируем поглощенную энергию для каждого сегмента нулем
for x in np.arange(minX, maxX + segmentDelta, segmentDelta):
    for z in np.arange(minZ, maxZ + segmentDelta, segmentDelta):
        segment_energy[(x, z)] = 0.0

n_photons = 150
n_steps = 15

wave_length = 337 / (10**6)

def isReflected(z):
    return z < 0

def isPassed(z):
  global tissue
  layerZ = 0
  for layer in tissue[:-1]:
      layerZ += layer['d']
      if (z < layerZ):
          return False

  return True

def photon_location(z):
    global tissue
    # z КООРДИНАТА < 0 или > maxD должна обрабатываться ОТДЕЛЬНО
    # данный if и tissue[-1] это заглушки!
    if z < 0:
        return tissue[0]

    # Иначе, ищем слой, в котором находится точка с координатой z
    layerZ = 0
    for layer in tissue:
        layerZ += layer['d']
        if z < layerZ:
            return layer

    return tissue[-1]

def generate_theta(g):
    # Определение обратной функции распределения для фазовой функции Хени-Гринштейна
    def inverse_hg_phase_function(g, cos_val):
          numerator = 1 + g**2 - ((1 - g**2) / (1 - g + 2*g*cos_val))**2
          denominator = 2 * g

          return np.arccos(numerator / denominator)

    u = np.random.uniform(0, 1)

    # Вычислить угол theta, используя обратную функцию распределения
    return inverse_hg_phase_function(g, u)

def calculateNewV(theta, phi, vect):
  cosTh = np.cos(theta)
  sinTh = np.sin(theta)

  cosPhi = np.cos(phi)
  sinPhi = np.sin(phi)

  _Vx, _Vy, _Vz = vect

  if np.abs(_Vz) == 1 and _Vx == 0 and _Vy == 0:
      Vx = sinTh * cosPhi
      Vy = sinTh * sinPhi
      Vz = cosPhi * _Vz

      return (Vx, Vy, Vz)

  zDif = np.sqrt(1 - _Vz**2)

  Vx = (_Vx * cosTh) + (sinTh/zDif) * (_Vx * _Vz * cosPhi - _Vy * sinPhi)
  Vy = (_Vy * cosTh) + (sinTh/zDif) * (_Vy * _Vz * cosPhi + _Vx * sinPhi)
  Vz = (_Vz * cosTh) - (sinTh * cosPhi * zDif)

  return Vx, Vy, Vz

# ua + us
def generate_s(u):
    wave_cm = np.random.exponential(scale=u);

    return wave_cm * wave_length

def getNewCoords(coords, vector, layer):
  s = generate_s(layer['ua'] + layer['us'])
  Vx, Vy, Vz = vector
  _x, _y, _z = coords

  x = _x + Vx * s
  y = _y + Vy * s
  z = _z + Vz * s

  return (x, y, z)


W_border = 0.2
m = 10

alivePhotons = 0;
reflected = 0;
passed = 0;

for phot in range(n_photons):
  coords = (0, 0, 0)

#   theta = generate_theta(tissue[0]['g'])
#   phi = np.random.uniform(0, 2 * math.pi)
#   vector = calculateNewV(theta, phi, (0, 0, 1))
  vector = (0, 0, 1)

  W = 1
  destroyed = False

  for step in range(n_steps):
    # Расчет данных на текущей позиции
    destroyed = False
    xCoord = coords[0]
    zCoord = coords[2]

      # Uncomment in case once reflected photons are not moving anymore
    if zCoord < 0 or zCoord > maxD:
        break

    layer = photon_location(zCoord)

    uSum = layer['ua'] + layer['us'];
    destroyProbability = layer['ua'] / uSum
    liveProbability = random.random()

    if (liveProbability < destroyProbability):
        destroyed = True
        break;

    W_delta = W * layer['ua']/uSum
    W -= W_delta
    if (W <= W_border):
        W_destroyProbability = random.random()
        if (W_destroyProbability <= 1/m):
            W *= m
        else:
            destroyed = True
            break;


    # Считаем поглощенную энергию
    xNorm = xCoord * 1000
    zNorm = zCoord * 1000

    xNorm -= xNorm % segmentDelta
    zNorm -= zNorm % segmentDelta

    if (zNorm >= minZ and zNorm <= maxZ and np.abs(xNorm) <= maxX):
      segment_energy[(xNorm, zNorm)] += W_delta


    # Расчет новой позиции для след итерации
    coords = getNewCoords(coords, vector, layer)

    theta = generate_theta(layer['g'])
    phi = np.random.uniform(0, 2*math.pi)
    vector = calculateNewV(theta, phi, vector)

  alivePhotons += not destroyed

  if (not destroyed):
      zCoord = coords[2]
      reflected += isReflected(zCoord)
      passed += isPassed(zCoord)

print(f"passed: {round(passed / n_photons * 100, 2)}%")
print(f"reflected {round(reflected / n_photons * 100, 2)}%")
# print(f"alivePhotons: {alivePhotons}, {round(alivePhotons / n_photons * 100, 3)}%")

# Построение графика
x_coords = np.arange(minX, maxX, segmentDelta)  # Ваш диапазон координат x
z_coords = np.arange(minZ, maxZ, segmentDelta)  # Ваш диапазон координат z

energy_values = np.zeros((len(z_coords), len(x_coords)))  # Создаем массив для значений энергии

for x in range(minX, maxX, segmentDelta):
    for z in range(minZ, maxZ, segmentDelta):
        xPos = int((x + xDelta) / segmentDelta)
        zPos = int(z / segmentDelta)

        value = segment_energy.get((x, z), 0)

        energy_values[zPos, xPos] = value  # Получаем значение из словаря, если есть


_vmaxVal = max(np.median(list(filter(lambda x: x > 0, segment_energy.values()))) * 3, 0.01);
plt.imshow(energy_values, cmap='hot', origin='upper', extent=[minX / normFactor, maxX / normFactor, maxZ / normFactor, minZ / normFactor], vmax=_vmaxVal, vmin=0)
plt.colorbar(label='Energy')  # Добавляем цветовую шкалу
plt.xlabel('X')  # Подпись оси x
plt.ylabel('Z')  # Подпись оси z
plt.title('Распределение поглощенной энергии')  # Заголовок графика
plt.show()
