(ns recommender-systems.core)

;; Based on 'Recommender Systems: An Introduction' by 
;; Jannach, Zanker, Felfernig, and Friedrich.

(def rating-data
  {:alice [5.0 3.0 4.0 4.0]
   :user1 [3.0 1.0 2.0 3.0 3.0]
   :user2 [4.0 3.0 4.0 3.0 5.0]
   :user3 [3.0 3.0 1.0 5.0 4.0]
   :user4 [1.0 5.0 5.0 2.0 1.0]})

(defn mean
  [xs]
  (/ (reduce + xs) (count xs)))

(defn pearson-numerator
  [userA userB]
  (let [zipped (map vector userA userB)]
    (apply +
           (map (fn [[a b]]
                  (* (- a (mean userA))
                     (- b (mean userB))))
                zipped))))

(defn pearson-denom
  [user-vals]
  (Math/sqrt (apply + 
                    (map (fn [a] 
                           (Math/pow (- a (mean user-vals)) 2))
                         user-vals))))

(defn pearson-similarity
  "An implementation of Pearson's correlation coefficient for
   calculating the similarity between two users based on a set of 
   product ratings for each user, on a scale of 1 (highly similar)
   to -1 (highly dissimilar)."
  [userA userB]
  (let [numerator (pearson-numerator userA userB)]
    (/ numerator (* (pearson-denom userA)
                    (pearson-denom (take (count userA) userB))))))

(defn nearest-neighbors
  [n target other-users]
  (take n (sort > (map (partial pearson-similarity target) other-users))))

(defn predict-rating
  "Predicts a user's rating of an unrated product N, where user-vals
   is a vector of the user's previous ratings for products and 
   neighbor-vals is a vector containing the product ratings for
   the target user's nearest neighbors (based on Pearson's correlation coefficient)."
  [user-vals neighbor-vals product]
  (let [ra (mean user-vals)]
    (+ ra
       (/ (apply + (map (fn [neighbor]
                          (* (pearson-similarity user-vals neighbor)
                             (- (nth neighbor product) 
                                (mean neighbor))))
                        neighbor-vals))
          (apply + (map (partial pearson-similarity user-vals) neighbor-vals))))))
