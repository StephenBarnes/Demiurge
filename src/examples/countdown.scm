; Checks that only the right branches of if-statements have side effects
(begin
	(set countdown
		(lambda (x)
			(if (= x 0)
				(print "liftoff")
				(begin
					(print x "...")
					(countdown (- x 1))))))
	(countdown 10))

