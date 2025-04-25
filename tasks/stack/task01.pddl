(define (problem skill-put-blocks-bin)
  (:domain tabletop-skill)
  (:objects
    red blue green - block   ; The three colored blocks.
  )
  (:init
    (arm-free)                    ; declarations
    (top-clear red)               ; Red block is initially on table
    (top-clear blue)              ; Blue block is initially on table
    (top-clear green)             ; Green block is initially on table
    (red red)
    (blue blue)                   
    (green green)                 
  )
  (:goal (and
           (on-block red green)      ; Red block must be in bin
           (on-block green blue)    ; Blue block must be in bin
  ))
)
