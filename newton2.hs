-- function f
f :: Double -> Double
f x = sqrt x * exp x - 1

-- gradient of f
f' :: Double -> Double
f' x = exp x * (0.5 * x ** (-0.5) + x ** 1.5)

-- Newton map
g :: Double -> Double
g x = x - (f x / f' x)

-- initial approximation
x0 :: Double
x0 = 1.0

-- list of approximations
xs :: [Double]
xs = iterate g x0

newton2 :: [Double]
newton2 = take 10 xs
