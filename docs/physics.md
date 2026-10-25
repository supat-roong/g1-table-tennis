# Ball and contact physics

## Ball

| Quantity | Value | Randomised range |
|---|---|---|
| Radius, mass | 20 mm, 2.7 g | fixed |
| Drag `kd` (a = -kd·\|v\|·v) | 0.126 1/m (Cd 0.45) | 0.10 to 0.15 |
| Magnus `km` (a = km·(w × v)) | 2.8e-3 | 1.8e-3 to 3.8e-3 |
| Spin decay | 5% per second | fixed |

## Surfaces

| Surface | Restitution | Friction | Randomised |
|---|---|---|---|
| Table | 0.88 (30 cm drop rebounds about 23 cm) | 0.25 | e 0.85 to 0.91, mu 0.15 to 0.35 |
| Rubber | 0.83 | 0.6 | e 0.75 to 0.90, mu 0.35 to 0.90 |

## Bounce model

A hollow sphere (I = 2/3·m·r²) hits a surface with normal n. The normal
impulse reverses the approach speed scaled by e. The tangential impulse is the
one that would stop the contact point slipping, capped at mu times the normal
impulse. Under the cap the ball leaves rolling; at the cap it leaves still
sliding. That is what makes topspin kick forward off the table and backspin
check, and what lets a rubbing stroke put spin on the ball.

## Timing

| Loop | Rate |
|---|---|
| Policy | 50 Hz |
| MuJoCo (robot) | 250 Hz |
| Ball, blade pose interpolated | 1 kHz |

## Serves

A serving machine starts every rally, alternating which robot receives.

| Parameter | Range |
|---|---|
| Machine position | x 1.5 to 1.8 m, y ±0.4 m, z 0.85 to 1.05 m |
| First bounce, receiver's half | x -1.30 to -0.95 m, y -0.5 to 0.35 m |
| Flight to bounce | 0.42 to 0.6 s |
| Topspin / backspin | -250 to 120 rad/s |
| Sidespin | ±80 rad/s |

Training starts with slow, nearly spinless serves (level 0: spin x0.2, no
sidespin, flights 0.10 to 0.15 s longer) and raises the level by 0.25 each time
40% of balls are hit for three logs in a row.

## Returns

Each robot's planner aims its return at the partner, in the partner's frame:

| Aim | Share | Partner meets the ball at | Lands at (hitter's frame) |
|---|---|---|---|
| Forehand | 65% | y -0.75 to -0.25 m | x 0.85 to 1.10 m, y 0.10 to 0.55 m |
| Backhand | 25% | y -0.05 to 0.40 m | x 1.00 to 1.20 m, y -0.45 to -0.05 m, 0.30 m over the net |
| Wide | 10% | y -1.05 to -0.80 m, beyond standing reach | x 0.30 to 0.90 m, y 0.40 to 0.74 m |

| Plane | Value |
|---|---|
| Hitting plane (ball centre) | x = -1.40 m, just behind the robot's end line (-1.37 m); z 0.78 to 1.00 m |
| Balls a robot goes for | y -1.15 to 0.90 m, z 0.70 to 1.20 m |
| Backhand | planned contact less than 0.15 m to the right of the robot's spot |

With Success Guided Sampling (the default), half the serves and half the
returns are instead picked among 2,048 serves and 36 return configurations
(aim x 3 depths x 3 widths) by how close their recent success is to 50%.
