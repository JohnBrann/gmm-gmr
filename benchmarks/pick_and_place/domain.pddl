(define (domain tabletop-skill)
  (:requirements :strips :typing)
  (:types block table bin)
  (:predicates
    (block ?b)                         ; ?b is a block.
    (table ?t)                         ; ?t is a table.
    (bin ?x)                           ; ?x is a bin.
    (on-table ?b ?t)                   ; block ?b is on table ?t.
    (in-bin ?b ?x)                     ; block ?b is in bin ?x.
    (arm-free)                         ; the robot arm is free.
    (holding ?b)                       ; the arm is currently holding block ?b.
  )

  ;; PICK: picks up a block from the table
  (:action pick
    :parameters (?b - block ?t - table)
    :precondition (and (on-table ?b ?t) (arm-free))
    :effect (and (holding ?b)
                 (not (arm-free))
                 (not (on-table ?b ?t)))
  )

  ;; DROP-RED: drops the red block into the bin
  (:action drop-red
    :parameters (?red - block ?x - bin)
    :precondition (holding ?red)
    :effect (and (in-bin ?red ?x)
                 (arm-free)
                 (not (holding ?red)))
  )

  ;; DROP-BLUE: drops the blue block into the bin
  ;; red block must already be in the bin
  (:action drop-blue
    :parameters (?blue - block ?x - bin)
    :precondition (and (holding ?blue)
                       (in-bin red ?x))
    :effect (and (in-bin ?blue ?x)
                 (arm-free)
                 (not (holding ?blue)))
  )

  ;; DROP-GREEN: drops the green block into the bin.
  ;; blue block must already be in the bin
  (:action drop-green
    :parameters (?green - block ?x - bin)
    :precondition (and (holding ?green)
                       (in-bin blue ?x))
    :effect (and (in-bin ?green ?x)
                 (arm-free)
                 (not (holding ?green)))
  )
)
