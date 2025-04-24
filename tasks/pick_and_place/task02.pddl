(define (problem skill-put-blocks-bin)
  (:domain tabletop-skill)
  (:objects
    red blue green orange purple - block   ; The colored blocks.
    table1 - table           ; The tabletop.
    bin1 - bin               ; The bin on the right side.
  )
  (:init
    (table table1)                ; declarations
    (bin bin1)                    
    (arm-free)                    
    (on-table red table1)         ; Red block is initially on table
    (on-table blue table1)        ; Blue block is initially on table
    (on-table green table1)       ; Green block is initially on table
    (on-table orange table1)       ; Green block is initially on table
    (on-table purple table1)       ; Green block is initially on table
    (red red)                    
    (blue blue)                   
    (green green)   
    (orange orange)
    (purple purple)              
  )
  (:goal (and
           (in-bin red bin1)      ; Red block must be in bin
           (in-bin blue bin1)     ; Blue block must be in bin
           (in-bin green bin1)    ; Green block must be in bin
           (in-bin orange bin1)    ; Orange block must be in bin
           (in-bin purple bin1)    ; Purple block must be in bin
  ))
)
