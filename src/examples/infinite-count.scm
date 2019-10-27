; demonstrates the set_global command, by using a global `i` to count from 0 upwards
(begin
	(set i 0)
	(infinite_loop
		(lambda ()
			(begin
				(set_global i (+ i 1))
				(print i)))))
