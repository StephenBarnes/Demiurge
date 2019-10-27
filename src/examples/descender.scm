; Program that descends forever down to lower interpreters, by modifying each to do the same

; First, define a function that prints out depth and then runs itself one level deeper
(set_universal descender
               (floatinglambda () ; note that this MUST be a floating lambda, or we'll capture 'DEPTH'
                 (begin
                   (print "DESCENDER IS AT DEPTH" DEPTH)
                   (joots (descender)))))
; Now run it
(descender)

; Gets down 70 layers before default recursion depth limit is exceeded
