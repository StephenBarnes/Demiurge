; example of how the FINISH-BEFORE directive works

(print "before redefining tokenization to use curly braces")

; define a tokenization function that uses curly braces
; this should return a list of tokens, then the remaining text (which we'll just make None)
(set_universal new_tokenize 
    (lambda (s interpreter) (begin
        ; first remove comments
        (set s
            (remove_comments s INTERPRETER))
        ; then parse into tokens
        (set tokens
            (regex_findall (+ "\{|\}|[^\'{}\s]+|\'[^\']*\'") s))
        ; then replace the {} with (), to plug into parse_tokens properly
        (set tokens
            (map
                (lambda (x) (if
                    (= x "{")
                        "("
                        (if (= x "}")
                            ")"
                            x)))
                tokens))
        (` tokens #f))))

; now assign interpreter's tokenize to the new tokenization function
(joots (set tokenize new_tokenize))

;FINISH-BEFORE

{print "after redefining tokenization to use curly braces"}
{print {+ 1 {* 2 3}}}
