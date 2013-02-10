import Data.Ratio

-- function f
f :: Rational -> Rational
f x = x * x - 2

-- gradient of f
f' :: Rational -> Rational
f' x = 2 * x

-- Newton map
g :: Rational -> Rational
g x = x - (f x / f' x)

-- initial approximation
x0 :: Rational
x0 = 1 % 1

-- list of approximations
xs :: [Rational]
xs = iterate g x0
