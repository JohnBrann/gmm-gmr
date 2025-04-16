; This is a new custom planner for a pick and place task

(define (domain tabletop-skill)
  (:requirements :strips :typing)
  (:types block table bin)
  (:predicates
    (block ?b)                         ; ?b is a block
    (table ?t)                         ; ?t is a table
    (bin ?x)                           ; ?x is a bin
    (on-table ?b ?t)                   ; block ?b is on table ?t
    (in-bin ?b ?x)                     ; block ?b is in bin ?x
    (arm-free)                         ; the robot arm is free and not holding anything
    (holding ?b)                       ; arm is currently holding block ?b.
    (red ?b)                           
    (blue ?b)                          
    (green ?b)                         
  )

  ;; PICK: picks up a block from the table.
  (:action pick
    :parameters (?b - block ?t - table)
    :precondition (and (on-table ?b ?t) (arm-free))
    :effect (and (holding ?b)
                 (not (arm-free))
                 (not (on-table ?b ?t)))
  )



  ;; PLACE: places a block into the bin on the table
  (:action place
    :parameters (?b - block ?x - bin)
    :precondition (and (holding ?b))
    :effect (and (in-bin ?b ?x)
                 (arm-free)
                 (not (holding ?b)))
  )
)