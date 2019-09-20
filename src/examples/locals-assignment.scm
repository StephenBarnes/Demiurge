; Try to assign variable by inserting it into context, like Python's `locals()['x'] = 10`
(begin
	(set_index (get_current_context) "x" 10)
	(print x))
