(begin
	(set curriedplus
		(lambda (x)
			(lambda (y)
				(lambda (z)
					(+ x y z)))))
	(set plus2 (curriedplus 2))
	(set plus5 (plus2 3))
	(plus5 10)) ; correctly prints 15
