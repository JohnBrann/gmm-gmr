; Problem file: skill-push-blocks-table

(define (problem skill-push-blocks-table)
  (:domain tabletop-push)
  (:objects
    red   blue   green  - block   ; the three colored blocks
    table1              - table   ; the single table
  )
  (:init
    (block  red)   (red   red)
    (block  blue)  (blue  blue)
    (block  green) (green green)
    (table  table1)
  )
  (:goal (and
    (on-table green table1)
    (on-table blue  table1)
    (on-table red   table1)
  ))
)
