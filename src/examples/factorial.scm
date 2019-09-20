(begin
	(set factorial
		(lambda (x)
			(if (= x 0)
				1
				(* x (factorial (- x 1))))))
	(factorial 5))

