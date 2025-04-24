(define (problem skill-put-blocks-bin)
  (:domain tabletop-skill)
  (:objects
    red blue green - block   ; The three colored blocks.
    table1 - table           ; The tabletop.
    bin1 - bin               ; Red bin.
    bin2 - bin               ; Green bin.
    bin3 - bin               ; Blue bin.
  )
  (:init
    (table table1)                ; declarations
    (bin bin1)
    (bin bin2)
    (bin bin3)
    (arm-free)                    
    (on-table red table1)         ; Red block is initially on table
    (on-table blue table1)        ; Blue block is initially on table
    (on-table green table1)       ; Green block is initially on table
    (red red)
    (blue blue)                   
    (green green)                 
  )
  (:goal (and
           (in-bin red bin1)      ; Red block must be in bin
           (in-bin green bin2)     ; Blue block must be in bin
           (in-bin blue bin3)    ; Green block must be in bin
  ))
)
