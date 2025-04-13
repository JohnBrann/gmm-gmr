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

  ;; DROP-RED: drops the red block into the bin.
  ;; only possible when holding a red block
  (:action drop-red
    :parameters (?r - block ?x - bin)
    :precondition (and (holding ?r)
                       (red ?r))
    :effect (and (in-bin ?r ?x)
                 (arm-free)
                 (not (holding ?r)))
  )

  ;; DROP-BLUE: drops the blue block into the bin.
  ;; only possible when holding a blue block and red is already in the bin
  (:action drop-blue
    :parameters (?b - block ?x - bin)
    :precondition (and (holding ?b)
                       (blue ?b)
                       (in-bin red ?x))
    :effect (and (in-bin ?b ?x)
                 (arm-free)
                 (not (holding ?b)))
  )

  ;; DROP-GREEN: drops the green block into the bin.
  ;; only possible when holding a green block and blue is already in the bin
  (:action drop-green
    :parameters (?b - block ?x - bin)
    :precondition (and (holding ?b)
                       (green ?b)
                       (in-bin blue ?x))
    :effect (and (in-bin ?b ?x)
                 (arm-free)
                 (not (holding ?b)))
  )
)
