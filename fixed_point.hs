-- function f
f :: Double -> Double
f x = exp(-2 * x)

-- initial approximation
x0 :: Double
x0 = 1.0

-- list of approximations
xs :: [Double]
xs = iterate f x0
