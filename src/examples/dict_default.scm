; redefines `index` operation to return a default value when not found
(set old-index index)
(set index
	(lambda (idx list)
		(if (in idx list)
			(old-index idx list)
			"not found")))
(set D (dict))
(print (index "x" D)) ; "not found"
(set_index D "x" 3)
(print (index "x" D)) ; 3
(print (index "y" D)) ; "not found"
