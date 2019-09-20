; demonstrates that lambda-functions create a local sub-context of the context they're defined in
(begin
    (set f
        (lambda (x)
            (get_current_context)))
    (f 3))
