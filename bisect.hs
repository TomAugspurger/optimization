-- function f
f :: Double -> Double
f x = sqrt x * exp x - 1

sign :: Double -> Integer
sign x = if x < 0 then -1 else 1

mid :: (Double, Double) -> Double
mid x = (fst x + snd x) / 2

bisection :: (Double, Double) -> (Double, Double)
bisection a =
    if sign(f (mid a)) == sign(f (fst a)) then (mid a, snd a) else (fst a, mid a)


x0 :: (Double, Double)
x0 = (0.1, 1.0)

xs :: [(Double, Double)]
xs = iterate bisection x0
